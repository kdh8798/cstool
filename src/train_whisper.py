import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_from_disk, concatenate_datasets
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "whisper_lora_manual")

MODEL_NAME = "openai/whisper-small"


def collate_fn(batch, pad_token_id=50257):
    input_features = torch.tensor(
        [item["input_features"] for item in batch],
        dtype=torch.float32
    )

    labels = [item["labels"] for item in batch]
    max_len = max(len(x) for x in labels)

    padded_labels = []
    for x in labels:
        padded = x + [-100] * (max_len - len(x))
        padded_labels.append(padded)

    labels = torch.tensor(padded_labels, dtype=torch.long)

    return {
        "input_features": input_features,
        "labels": labels,
    }


def main():
    ko_path = os.path.join(DATA_DIR, "commonvoice_ko_train")
    ru_path = os.path.join(DATA_DIR, "commonvoice_ru_train")

    ko_ds = load_from_disk(ko_path)
    ru_ds = load_from_disk(ru_path)

    print("KO columns before cleanup:", ko_ds.column_names)
    print("RU columns before cleanup:", ru_ds.column_names)

    keep_columns = ["input_features", "labels"]

    ko_remove_cols = [c for c in ko_ds.column_names if c not in keep_columns]
    ru_remove_cols = [c for c in ru_ds.column_names if c not in keep_columns]

    if ko_remove_cols:
        ko_ds = ko_ds.remove_columns(ko_remove_cols)
    if ru_remove_cols:
        ru_ds = ru_ds.remove_columns(ru_remove_cols)

    print("KO columns after cleanup:", ko_ds.column_names)
    print("RU columns after cleanup:", ru_ds.column_names)

    dataset = concatenate_datasets([ko_ds, ru_ds]).shuffle(seed=42)

    print("Dataset columns:", dataset.column_names)
    print("Dataset size:", len(dataset))

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Start training...")

    model.train()
    num_epochs = 3

    for epoch in range(num_epochs):
        total_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True
        )

        for step, batch in enumerate(progress_bar):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            decoder_input_ids = labels[:, :-1].clone()
            decoder_labels = labels[:, 1:].clone()

            decoder_input_ids[decoder_input_ids == -100] = processor.tokenizer.pad_token_id

            outputs = model(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids
            )

            logits = outputs.logits

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                decoder_labels.reshape(-1)
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_value = loss.item()
            total_loss += loss_value

            progress_bar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "avg_loss": f"{(total_loss / (step + 1)):.4f}"
            })

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch + 1}] Average Loss: {avg_loss:.4f}")

        epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    print(f"Training complete. Final model saved to: {final_dir}")


if __name__ == "__main__":
    main()
