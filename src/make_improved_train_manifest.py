import random
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

KO_AUG_CSV = BASE_DIR / "data" / "processed" / "ko_augmented" / "metadata.csv"
RU_DIR = BASE_DIR / "data" / "raw" / "cv_ru"
CODESWITCH_CSV = BASE_DIR / "data" / "processed" / "codeswitch_eval" / "metadata.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "improved_train"
OUTPUT_CSV = OUTPUT_DIR / "metadata.csv"

RANDOM_SEED = 42

MAX_KO = 2200
MAX_RU = 2000
MAX_CS = 1000


def load_ru_samples():
    test_path = RU_DIR / "test.tsv"
    validated_path = RU_DIR / "validated.tsv"

    if test_path.exists():
        tsv_path = test_path
    elif validated_path.exists():
        tsv_path = validated_path
    else:
        raise FileNotFoundError("cv_ru 안에서 test.tsv 또는 validated.tsv를 찾지 못했습니다.")

    df = pd.read_csv(tsv_path, sep="\t")

    rows = []

    for _, row in df.iterrows():
        audio_path = RU_DIR / "clips" / row["path"]

        if audio_path.exists():
            rows.append({
                "audio_path": str(audio_path),
                "sentence": str(row["sentence"]).strip(),
                "lang": "ru"
            })

    return pd.DataFrame(rows)


def load_codeswitch_samples():
    if not CODESWITCH_CSV.exists():
        raise FileNotFoundError(
            f"코드스위칭 metadata를 찾지 못했습니다: {CODESWITCH_CSV}\n"
            "먼저 python src\\make_codeswitch_data.py 를 실행하세요."
        )

    df = pd.read_csv(CODESWITCH_CSV)

    rows = []

    clips_dir = BASE_DIR / "data" / "processed" / "codeswitch_eval" / "clips"

    for _, row in df.iterrows():
        audio_path = clips_dir / row["path"]

        if audio_path.exists():
            rows.append({
                "audio_path": str(audio_path),
                "sentence": str(row["sentence"]).strip(),
                "lang": "mixed"
            })

    return pd.DataFrame(rows)


def sample_df(df, max_count, random_state=42):
    if len(df) <= max_count:
        return df

    return df.sample(n=max_count, random_state=random_state)


def main():
    random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading KO augmented dataset...")
    if not KO_AUG_CSV.exists():
        raise FileNotFoundError(
            f"한국어 증강 metadata를 찾지 못했습니다: {KO_AUG_CSV}\n"
            "먼저 python src\\make_augmented_ko_data.py 를 실행하세요."
        )

    df_ko = pd.read_csv(KO_AUG_CSV)
    df_ko = sample_df(df_ko, MAX_KO, RANDOM_SEED)
    print(f"KO selected: {len(df_ko)}")

    print("Loading RU dataset...")
    df_ru = load_ru_samples()
    df_ru = sample_df(df_ru, MAX_RU, RANDOM_SEED)
    print(f"RU selected: {len(df_ru)}")

    print("Loading code-switch dataset...")
    df_cs = load_codeswitch_samples()
    df_cs = sample_df(df_cs, MAX_CS, RANDOM_SEED)
    print(f"Code-switch selected: {len(df_cs)}")

    df_final = pd.concat(
        [
            df_ko[["audio_path", "sentence", "lang"]],
            df_ru[["audio_path", "sentence", "lang"]],
            df_cs[["audio_path", "sentence", "lang"]],
        ],
        ignore_index=True
    )

    df_final = df_final.sample(frac=1.0, random_state=RANDOM_SEED)

    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nSaved improved train manifest:")
    print(OUTPUT_CSV)
    print(f"Total samples: {len(df_final)}")


if __name__ == "__main__":
    main()
