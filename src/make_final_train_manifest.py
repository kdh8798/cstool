import random
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

BALANCED_CSV = BASE_DIR / "data" / "processed" / "balanced_train" / "metadata.csv"
CODESWITCH_CSV = BASE_DIR / "data" / "processed" / "codeswitch_eval" / "metadata.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "final_train"
OUTPUT_CSV = OUTPUT_DIR / "metadata.csv"

RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading balanced dataset...")
    df_balanced = pd.read_csv(BALANCED_CSV)
    print(f"Balanced samples: {len(df_balanced)}")

    print("Loading code-switch dataset...")
    df_cs = pd.read_csv(CODESWITCH_CSV)
    print(f"Code-switch samples: {len(df_cs)}")

    # code-switch 데이터 변환
    cs_rows = []

    for _, row in df_cs.iterrows():
        audio_path = BASE_DIR / "data" / "processed" / "codeswitch_eval" / "clips" / row["path"]

        cs_rows.append({
            "audio_path": str(audio_path),
            "sentence": row["sentence"],
            "lang": "mixed"
        })

    df_cs_final = pd.DataFrame(cs_rows)

    # 합치기
    df_final = pd.concat([df_balanced, df_cs_final], ignore_index=True)

    df_final = df_final.sample(frac=1.0, random_state=RANDOM_SEED)

    print(f"\nFinal dataset size: {len(df_final)}")

    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved final train dataset to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
