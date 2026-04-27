import random
from pathlib import Path

import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent

KO_DIR = BASE_DIR / "data" / "raw" / "cv_ko"
RU_DIR = BASE_DIR / "data" / "raw" / "cv_ru"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "codeswitch_eval"
OUTPUT_CLIPS_DIR = OUTPUT_DIR / "clips"
OUTPUT_METADATA = OUTPUT_DIR / "metadata.csv"

TARGET_SR = 16000
NUM_SAMPLES = 200

RANDOM_SEED = 42


# Common Voice TSV 로드
def load_commonvoice_samples(lang_dir: Path):
    test_path = lang_dir / "test.tsv"
    validated_path = lang_dir / "validated.tsv"

    if test_path.exists():
        tsv_path = test_path
    elif validated_path.exists():
        tsv_path = validated_path
    else:
        raise FileNotFoundError(f"{lang_dir} 안에서 test.tsv 또는 validated.tsv 를 찾지 못했습니다.")

    df = pd.read_csv(tsv_path, sep="\t")

    if "path" not in df.columns or "sentence" not in df.columns:
        raise ValueError(f"{tsv_path} 에 path 또는 sentence 컬럼이 없습니다.")

    samples = []

    for _, row in df.iterrows():
        audio_path = lang_dir / "clips" / row["path"]

        if audio_path.exists():
            samples.append({
                "audio_path": audio_path,
                "sentence": str(row["sentence"]).strip()
            })

    return samples


# 오디오 로드
def load_audio(audio_path: Path):
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    return audio


# 코드스위칭 음성 생성
def make_codeswitch_audio(audio_a, audio_b, silence_sec=0.25):
    silence = np.zeros(int(TARGET_SR * silence_sec), dtype=np.float32)
    mixed = np.concatenate([audio_a, silence, audio_b])
    return mixed


# 메인
def main():
    random.seed(RANDOM_SEED)

    OUTPUT_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Korean samples...")
    ko_samples = load_commonvoice_samples(KO_DIR)
    print(f"KO samples: {len(ko_samples)}")

    print("Loading Russian samples...")
    ru_samples = load_commonvoice_samples(RU_DIR)
    print(f"RU samples: {len(ru_samples)}")

    metadata_rows = []

    for i in tqdm(range(NUM_SAMPLES), desc="Creating code-switching data"):
        pattern = "ko_ru" if i % 2 == 0 else "ru_ko"

        ko_sample = random.choice(ko_samples)
        ru_sample = random.choice(ru_samples)

        try:
            ko_audio = load_audio(ko_sample["audio_path"])
            ru_audio = load_audio(ru_sample["audio_path"])

            if pattern == "ko_ru":
                output_audio = make_codeswitch_audio(ko_audio, ru_audio)
                sentence = f"{ko_sample['sentence']} {ru_sample['sentence']}"
            else:
                output_audio = make_codeswitch_audio(ru_audio, ko_audio)
                sentence = f"{ru_sample['sentence']} {ko_sample['sentence']}"

            filename = f"cs_{i + 1:06d}.wav"
            output_path = OUTPUT_CLIPS_DIR / filename

            sf.write(str(output_path), output_audio, TARGET_SR)

            metadata_rows.append({
                "path": filename,
                "sentence": sentence,
                "pattern": pattern,
                "ko_source": str(ko_sample["audio_path"]),
                "ru_source": str(ru_sample["audio_path"]),
                "ko_sentence": ko_sample["sentence"],
                "ru_sentence": ru_sample["sentence"],
            })

        except Exception as e:
            print(f"[ERROR] sample {i}: {type(e).__name__}: {e}")

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Saved clips to: {OUTPUT_CLIPS_DIR}")
    print(f"Saved metadata to: {OUTPUT_METADATA}")
    print(f"Total generated: {len(metadata_rows)}")


if __name__ == "__main__":
    main()
