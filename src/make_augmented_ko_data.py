import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent

KO_DIR = BASE_DIR / "data" / "raw" / "cv_ko"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "ko_augmented"
OUTPUT_CLIPS_DIR = OUTPUT_DIR / "clips"
OUTPUT_CSV = OUTPUT_DIR / "metadata.csv"

TARGET_SR = 16000
RANDOM_SEED = 42


def load_ko_samples():
    test_path = KO_DIR / "test.tsv"
    validated_path = KO_DIR / "validated.tsv"

    if test_path.exists():
        tsv_path = test_path
    elif validated_path.exists():
        tsv_path = validated_path
    else:
        raise FileNotFoundError("cv_ko 안에서 test.tsv 또는 validated.tsv를 찾지 못했습니다.")

    df = pd.read_csv(tsv_path, sep="\t")

    if "path" not in df.columns or "sentence" not in df.columns:
        raise ValueError(f"{tsv_path}에 path 또는 sentence 컬럼이 없습니다.")

    samples = []

    for _, row in df.iterrows():
        audio_path = KO_DIR / "clips" / row["path"]

        if audio_path.exists():
            samples.append({
                "audio_path": audio_path,
                "sentence": str(row["sentence"]).strip()
            })

    return samples


def load_audio(audio_path: Path):
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    return np.asarray(audio, dtype=np.float32)


def speed_change(audio, rate: float):
    return librosa.effects.time_stretch(audio, rate=rate)


def add_noise(audio, noise_level=0.003):
    noise = np.random.normal(0, noise_level, size=audio.shape).astype(np.float32)
    augmented = audio + noise
    return np.clip(augmented, -1.0, 1.0)


def save_audio(audio, output_path: Path):
    sf.write(str(output_path), audio, TARGET_SR)


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    OUTPUT_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_ko_samples()
    print(f"Loaded KO samples: {len(samples)}")

    rows = []

    for idx, sample in enumerate(tqdm(samples, desc="Creating KO augmented data")):
        try:
            audio = load_audio(sample["audio_path"])
            sentence = sample["sentence"]

            variants = {
                "orig": audio,
                "speed095": speed_change(audio, 0.95),
                "speed105": speed_change(audio, 1.05),
                "noise": add_noise(audio),
            }

            for aug_type, aug_audio in variants.items():
                filename = f"ko_aug_{idx:06d}_{aug_type}.wav"
                output_path = OUTPUT_CLIPS_DIR / filename

                save_audio(aug_audio, output_path)

                rows.append({
                    "audio_path": str(output_path),
                    "sentence": sentence,
                    "lang": "ko",
                    "source": str(sample["audio_path"]),
                    "augmentation": aug_type,
                })

        except Exception as e:
            print(f"[ERROR] {sample['audio_path'].name} | {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Saved clips to: {OUTPUT_CLIPS_DIR}")
    print(f"Saved metadata to: {OUTPUT_CSV}")
    print(f"Total augmented samples: {len(rows)}")


if __name__ == "__main__":
    main()