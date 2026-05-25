import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# 설정
BASE_DIR = Path(__file__).resolve().parent.parent

KO_DIR = BASE_DIR / "data" / "raw" / "cv_ko"
RU_DIR = BASE_DIR / "data" / "raw" / "cv_ru"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "balanced_train"
OUTPUT_CSV = OUTPUT_DIR / "metadata.csv"

RANDOM_SEED = 42


# Common Voice 로드
def load_commonvoice(lang_dir: Path):
    test_path = lang_dir / "test.tsv"
    validated_path = lang_dir / "validated.tsv"

    if test_path.exists():
        tsv_path = test_path
    elif validated_path.exists():
        tsv_path = validated_path
    else:
        raise FileNotFoundError(f"{lang_dir} 에서 test.tsv 또는 validated.tsv 없음")

    df = pd.read_csv(tsv_path, sep="\t")

    if "path" not in df.columns or "sentence" not in df.columns:
        raise ValueError(f"{tsv_path} 컬럼 문제")

    samples = []

    for _, row in df.iterrows():
        audio_path = lang_dir / "clips" / row["path"]

        if audio_path.exists():
            samples.append({
                "audio_path": str(audio_path),
                "sentence": str(row["sentence"]),
                "lang": lang_dir.name  # cv_ko / cv_ru
            })

    return samples


# 메인
def main():
    random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Korean dataset...")
    ko_samples = load_commonvoice(KO_DIR)
    print(f"KO samples: {len(ko_samples)}")

    print("Loading Russian dataset...")
    ru_samples = load_commonvoice(RU_DIR)
    print(f"RU samples: {len(ru_samples)}")

    # 균형 맞추기
    target_size = min(len(ko_samples), len(ru_samples))

    print(f"\nBalancing datasets to {target_size} samples each")

    ko_selected = random.sample(ko_samples, target_size)
    ru_selected = random.sample(ru_samples, target_size)

    combined = ko_selected + ru_selected
    random.shuffle(combined)

    print(f"Total combined samples: {len(combined)}")

    # CSV 저장
    df = pd.DataFrame(combined)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nSaved balanced dataset:")
    print(f"{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
