import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import re
import json
import warnings
from pathlib import Path

import torch
import evaluate
import pandas as pd
import librosa
import numpy as np

from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging


# 로그 / 경고 숨기기
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


# 기본 설정
BASE_MODEL = "openai/whisper-small"

BASE_DIR = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = BASE_DIR / "data" / "raw"

LOCAL_DATASETS = {
    "ko": DATA_ROOT / "cv_ko",
    "ru": DATA_ROOT / "cv_ru",
}

OUTPUT_JSON = BASE_DIR / "results" / "baseline_wer_results.json"


# 로컬 Common Voice 로더
def load_local_commonvoice(
    lang_dir: Path,
    split_name: str = "test",
    max_samples: int | None = None
):
    split_path = lang_dir / f"{split_name}.tsv"

    if split_path.exists():
        tsv_path = split_path
    else:
        fallback = lang_dir / "validated.tsv"
        if fallback.exists():
            tsv_path = fallback
        else:
            raise FileNotFoundError(
                f"{lang_dir} 안에서 {split_name}.tsv 또는 validated.tsv 를 찾지 못했습니다."
            )

    df = pd.read_csv(tsv_path, sep="\t")

    required_cols = ["path", "sentence"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"{tsv_path} 에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}"
            )

    if max_samples is not None:
        df = df.head(max_samples)

    samples = []

    for _, row in df.iterrows():
        audio_path = lang_dir / "clips" / row["path"]

        if not audio_path.exists():
            continue

        samples.append({
            "audio_path": audio_path,
            "sentence": str(row["sentence"]),
        })

    return samples


# 오디오 로딩
def load_audio_file(audio_path: Path, target_sr: int = 16000):
    audio_array, sr = librosa.load(
        str(audio_path),
        sr=target_sr,
        mono=True
    )

    audio_array = np.asarray(audio_array, dtype=np.float32)

    return audio_array, sr


# 텍스트 정규화
def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    # 한국어, 러시아어, 숫자, 영어, 공백만 유지
    text = re.sub(r"[^\w\s가-힣а-яё]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 모델 로드
print("Loading processor...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

print("Loading baseline model...")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
model = model.to(DEVICE)
model.eval()

# max_new_tokens / max_length 중복 경고 방지
model.generation_config.max_length = None


# WER metric 준비
wer_metric = evaluate.load("wer")


# 추론 함수
@torch.no_grad()
def transcribe_audio(audio_array, sampling_rate=16000, lang_code=None):
    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(DEVICE)

    generate_kwargs = {
        "max_new_tokens": 128,
        "task": "transcribe",
    }

    if lang_code is not None:
        generate_kwargs["language"] = lang_code

    predicted_ids = model.generate(
        input_features,
        **generate_kwargs
    )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription


# 평가 실행
all_results = []
overall_references = []
overall_predictions = []

for lang_name, lang_dir in LOCAL_DATASETS.items():
    print(f"\n===== Evaluating baseline language: {lang_name} =====")

    samples = load_local_commonvoice(
        lang_dir=lang_dir,
        split_name="test",
        # 빠른 테스트용
        # max_samples=20
        max_samples=None
    )

    print(f"Loaded {len(samples)} samples from {lang_dir}")

    lang_references = []
    lang_predictions = []

    for idx, sample in enumerate(tqdm(samples, desc=f"Evaluating baseline {lang_name}")):
        try:
            audio_array, sampling_rate = load_audio_file(sample["audio_path"])
            reference = sample["sentence"]

            prediction = transcribe_audio(
                audio_array,
                sampling_rate,
                lang_code=lang_name
            )

            norm_ref = normalize_text(reference)
            norm_pred = normalize_text(prediction)

            lang_references.append(norm_ref)
            lang_predictions.append(norm_pred)

            overall_references.append(norm_ref)
            overall_predictions.append(norm_pred)

            all_results.append({
                "id": f"{lang_name}_{idx}",
                "lang": lang_name,
                "audio_path": str(sample["audio_path"]),
                "reference": reference,
                "prediction": prediction,
                "normalized_reference": norm_ref,
                "normalized_prediction": norm_pred,
            })

        except Exception as e:
            print(
                f"[ERROR] {lang_name}_{idx} | "
                f"{sample['audio_path'].name} | "
                f"{type(e).__name__}: {e}"
            )

    if len(lang_references) > 0:
        lang_wer = wer_metric.compute(
            predictions=lang_predictions,
            references=lang_references
        )
        print(f"Baseline {lang_name.upper()} WER: {lang_wer:.4f}")
    else:
        lang_wer = None
        print(f"Baseline {lang_name.upper()} WER: could not compute")

    all_results.append({
        "summary": True,
        "model": BASE_MODEL,
        "type": "baseline",
        "lang": lang_name,
        "wer": lang_wer,
        "num_samples": len(lang_references),
    })


# 전체 WER 계산
if len(overall_references) > 0:
    overall_wer = wer_metric.compute(
        predictions=overall_predictions,
        references=overall_references
    )
    print(f"\n===== BASELINE OVERALL WER: {overall_wer:.4f} =====")
else:
    overall_wer = None
    print("\n===== BASELINE OVERALL WER: could not compute =====")


# 결과 저장
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

final_output = {
    "model": BASE_MODEL,
    "type": "baseline",
    "overall_wer": overall_wer,
    "num_total_samples": len(overall_references),
    "results": all_results,
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print(f"Saved baseline results to {OUTPUT_JSON}")
