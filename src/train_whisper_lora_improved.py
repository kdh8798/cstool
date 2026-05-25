import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import json
import warnings
from pathlib import Path

import torch
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging

from peft import LoraConfig, get_peft_model


warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


# 1. 설정
BASE_MODEL = "openai/whisper-small"

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV = BASE_DIR / "data" / "processed" / "improved_train" / "metadata.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "whisper_lora_improved"
FINAL_DIR = OUTPUT_DIR / "final"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_SR = 16000

# 4070ti 기준 설정
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
EPOCHS = 5
LEARNING_RATE = 5e-5
MAX_LABEL_LENGTH = 448


# 2. Dataset
class ImprovedWhisperDataset(Dataset):
    def __init__(self, csv_path, processor):
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"학습 metadata를 찾지 못했습니다: {csv_path}\n"
                "먼저 python src\\make_improved_train_manifest.py 를 실행하세요."
            )

        self.df = pd.read_csv(csv_path)
        self.processor = processor

        required_cols = ["audio_path", "sentence", "lang"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"{csv_path} 에 '{col}' 컬럼이 없습니다.")

        self.samples = []

        for _, row in self.df.iterrows():
            audio_path = Path(row["audio_path"])

            if audio_path.exists():
                self.samples.append({
                    "audio_path": audio_path,
                    "sentence": str(row["sentence"]),
                    "lang": str(row["lang"]),
                })

        print(f"Loaded train samples: {len(self.samples)}")

        if len(self.samples) == 0:
            raise ValueError("학습 가능한 샘플이 0개입니다. audio_path 경로를 확인하세요.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        audio, sr = librosa.load(
            str(item["audio_path"]),
            sr=TARGET_SR,
            mono=True
        )
        audio = np.asarray(audio, dtype=np.float32)

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(
            item["sentence"],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=MAX_LABEL_LENGTH
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
            "lang": item["lang"],
        }


# 3. Collate 함수
def collate_fn(batch):
    input_features = torch.stack([x["input_features"] for x in batch])

    labels = [x["labels"] for x in batch]
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )

    return {
        "input_features": input_features,
        "labels": labels,
    }


# 4. 모델 / Processor 로드
print("Loading processor...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

# Whisper generation 관련 설정
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# 5. LoRA 설정
print("Applying LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model = model.to(DEVICE)


# 6. DataLoader
train_dataset = ImprovedWhisperDataset(TRAIN_CSV, processor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


# 7. Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# 8. 학습 루프
print("\nStart improved training...")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Grad accumulation steps: {GRAD_ACCUM_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Train CSV: {TRAIN_CSV}")
print(f"Output dir: {OUTPUT_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        input_features = batch["input_features"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_features=input_features,
            labels=labels
        )

        original_loss = outputs.loss
        loss = original_loss / GRAD_ACCUM_STEPS

        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += original_loss.item()
        avg_loss = total_loss / (step + 1)

        progress_bar.set_postfix({
            "loss": f"{original_loss.item():.4f}",
            "avg_loss": f"{avg_loss:.4f}"
        })

    # 남은 gradient 처리
    if len(train_loader) % GRAD_ACCUM_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    epoch_dir = OUTPUT_DIR / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving epoch {epoch} adapter to {epoch_dir}")
    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)

    epoch_log = {
        "epoch": epoch,
        "avg_loss": total_loss / len(train_loader),
        "num_samples": len(train_dataset),
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "train_csv": str(TRAIN_CSV),
    }

    with open(epoch_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(epoch_log, f, ensure_ascii=False, indent=2)

    print(f"Epoch {epoch} average loss: {epoch_log['avg_loss']:.4f}")


# 9. 최종 저장
FINAL_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nSaving final adapter to {FINAL_DIR}")
model.save_pretrained(FINAL_DIR)
processor.save_pretrained(FINAL_DIR)

final_log = {
    "base_model": BASE_MODEL,
    "train_csv": str(TRAIN_CSV),
    "output_dir": str(OUTPUT_DIR),
    "final_dir": str(FINAL_DIR),
    "num_samples": len(train_dataset),
    "batch_size": BATCH_SIZE,
    "grad_accum_steps": GRAD_ACCUM_STEPS,
    "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "target_sr": TARGET_SR,
}

with open(FINAL_DIR / "train_config.json", "w", encoding="utf-8") as f:
    json.dump(final_log, f, ensure_ascii=False, indent=2)

print("\nImproved training complete.")
print(f"Final LoRA saved to: {FINAL_DIR}")
