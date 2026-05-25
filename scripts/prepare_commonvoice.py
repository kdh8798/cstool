import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from datasets import load_dataset, Audio
from transformers import WhisperProcessor

from src.preprocessing import normalize_text, is_valid_sample
from src.dataset_utils import prepare_dataset

DATASET_NAME = "fsicoli/common_voice_17_0"
MODEL_NAME = "openai/whisper-small"

MAX_SAMPLES_PER_LANG = 500
OUTPUT_DIR = "data/processed"

def preprocess_commonvoice(lang_code: str, lang_name: str):
    print(f"\n=== Loading {lang_name} Common Voice ===")

    ds = load_dataset(
        DATASET_NAME,
        lang_code,
        split=f"train[:{MAX_SAMPLES_PER_LANG}]"
    )

    print(f"{lang_name} raw size:", len(ds))

    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    def normalize_batch(sample):
        sample["sentence"] = normalize_text(sample["sentence"])
        return sample

    ds = ds.map(normalize_batch)
    ds = ds.filter(is_valid_sample)

    print(f"{lang_name} filtered size:", len(ds))

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    ds = ds.map(
        lambda sample: prepare_dataset(sample, processor),
        remove_columns=ds.column_names
    )

    save_path = f"{OUTPUT_DIR}/commonvoice_{lang_code}"
    ds.save_to_disk(save_path)

    print(f"{lang_name} saved to: {save_path}")
    print(ds)
    return ds

if __name__ == "__main__":
    preprocess_commonvoice("ko", "Korean")
    preprocess_commonvoice("ru", "Russian")