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

from peft import LoraConfig, get_peft_model, TaskType


warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


# 설정
BASE_MODEL = "openai/whisper-small"

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV = BASE_DIR / "data" / "processed" / "final_train" / "metadata.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "whisper_lora_balanced"
FINAL_DIR = OUTPUT_DIR / "final"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_SR = 16000
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_LABEL_LENGTH = 448


# Dataset
class BalancedWhisperDataset(Dataset):
    def __init__(self, csv_path, processor):
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


# Collate 함수
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


# 모델 / Processor 로드
print("Loading processor...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# LoRA 설정
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


# DataLoader
train_dataset = BalancedWhisperDataset(TRAIN_CSV, processor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# 학습 루프
print("\nStart training...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for step, batch in enumerate(progress_bar):
        input_features = batch["input_features"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_features=input_features,
            labels=labels
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{avg_loss:.4f}"
        })

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
        "learning_rate": LEARNING_RATE,
    }

    with open(epoch_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(epoch_log, f, ensure_ascii=False, indent=2)

    print(f"Epoch {epoch} average loss: {epoch_log['avg_loss']:.4f}")


# 최종 저장
FINAL_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nSaving final adapter to {FINAL_DIR}")
model.save_pretrained(FINAL_DIR)
processor.save_pretrained(FINAL_DIR)

print("\nTraining complete.")
print(f"Final LoRA saved to: {FINAL_DIR}")
