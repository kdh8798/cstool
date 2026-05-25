"""
Dataset: Mozilla Common Voice
License: CC0 1.0 (Public Domain)
Source: https://commonvoice.mozilla.org
"""

import os
import sys
import pandas as pd
from datasets import Dataset
from transformers import WhisperProcessor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.preprocessing import normalize_text, is_valid_sample
from src.dataset_utils import prepare_dataset

MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def load_commonvoice_tsv(cv_dir: str, split: str = "train") -> Dataset:
    tsv_path = os.path.join(cv_dir, f"{split}.tsv")
    clips_dir = os.path.join(cv_dir, "clips")

    df = pd.read_csv(tsv_path, sep="\t")

    if "path" not in df.columns or "sentence" not in df.columns:
        raise ValueError(
            f"Expected 'path' and 'sentence' columns in {tsv_path}, got {df.columns.tolist()}"
        )

    df = df[["path", "sentence"]].copy()
    df["audio"] = df["path"].apply(lambda x: os.path.join(clips_dir, x))
    df = df.drop(columns=["path"])

    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds


def preprocess_local_commonvoice(
    cv_dir: str,
    lang_code: str,
    split: str = "train",
    max_samples: int | None = 100
):
    print(f"\n=== Loading local Common Voice: {lang_code} / {split} ===")
    ds = load_commonvoice_tsv(cv_dir, split=split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print("Raw size:", len(ds))

    def normalize_batch(sample):
        sample["sentence"] = normalize_text(sample["sentence"])
        return sample

    ds = ds.map(normalize_batch)
    ds = ds.filter(is_valid_sample)
    print("After validity filter:", len(ds))

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    ds = ds.map(
        lambda sample: prepare_dataset(sample, processor),
        remove_columns=ds.column_names
    )

    save_path = os.path.join(OUTPUT_DIR, f"commonvoice_{lang_code}_{split}")
    ds.save_to_disk(save_path)

    print(f"Saved to: {save_path}")
    print(ds)


if __name__ == "__main__":
    preprocess_local_commonvoice(
        r"C:\Github\cstool\data\raw\cv_ko",
        "ko",
        split="train",
        max_samples=100
    )

    preprocess_local_commonvoice(
        r"C:\Github\cstool\data\raw\cv_ru",
        "ru",
        split="train",
        max_samples=100
    )
