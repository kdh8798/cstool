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
from peft import PeftModel


# 로그 / 경고 숨기기
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


# 기본 설정
BASE_MODEL = "openai/whisper-small"

BASE_DIR = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LORA_PATH = BASE_DIR / "outputs" / "whisper_lora_manual" / "final"

CODESWITCH_DIR = BASE_DIR / "data" / "processed" / "codeswitch_eval"
CODESWITCH_CLIPS_DIR = CODESWITCH_DIR / "clips"
METADATA_CSV = CODESWITCH_DIR / "metadata.csv"

OUTPUT_JSON = BASE_DIR / "results" / "codeswitch_wer_results.json"


# 코드스위칭 metadata 로드
def load_codeswitch_samples(max_samples: int | None = None):
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"metadata.csv를 찾을 수 없습니다: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)

    required_cols = ["path", "sentence"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"{METADATA_CSV} 에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}"
            )

    if max_samples is not None:
        df = df.head(max_samples)

    samples = []

    for _, row in df.iterrows():
        audio_path = CODESWITCH_CLIPS_DIR / row["path"]

        if not audio_path.exists():
            continue

        samples.append({
            "audio_path": audio_path,
            "sentence": str(row["sentence"]),
            "pattern": str(row["pattern"]) if "pattern" in df.columns else "unknown",
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

    # 한국어, 러시아어, 영어, 숫자, 공백만 유지
    text = re.sub(r"[^\w\s가-힣а-яё]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 모델 로드
print("Loading processor...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.to(DEVICE)
model.eval()

# max_new_tokens / max_length 중복 경고 방지
model.generation_config.max_length = None


# WER metric
wer_metric = evaluate.load("wer")


# 추론 함수
@torch.no_grad()
def transcribe_audio(audio_array, sampling_rate=16000):
    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(DEVICE)

    # 코드스위칭에서는 language를 고정하지 않는 것이 좋음
    # ko 또는 ru로 고정하면 반대 언어 구간을 무시하거나 틀릴 가능성이 있음
    predicted_ids = model.generate(
        input_features,
        max_new_tokens=128,
        task="transcribe"
    )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription


# 평가 실행
all_results = []
references = []
predictions = []

print("\n===== Evaluating code-switching data =====")

samples = load_codeswitch_samples(
    max_samples=None
    # 빠른 테스트용
    # max_samples=20
)

print(f"Loaded {len(samples)} code-switching samples from {CODESWITCH_DIR}")

for idx, sample in enumerate(tqdm(samples, desc="Evaluating code-switching")):
    try:
        audio_array, sampling_rate = load_audio_file(sample["audio_path"])
        reference = sample["sentence"]

        prediction = transcribe_audio(audio_array, sampling_rate)

        norm_ref = normalize_text(reference)
        norm_pred = normalize_text(prediction)

        references.append(norm_ref)
        predictions.append(norm_pred)

        all_results.append({
            "id": f"cs_{idx}",
            "pattern": sample["pattern"],
            "audio_path": str(sample["audio_path"]),
            "reference": reference,
            "prediction": prediction,
            "normalized_reference": norm_ref,
            "normalized_prediction": norm_pred,
        })

    except Exception as e:
        print(
            f"[ERROR] cs_{idx} | "
            f"{sample['audio_path'].name} | "
            f"{type(e).__name__}: {e}"
        )


# 전체 코드스위칭 WER 계산
if len(references) > 0:
    codeswitch_wer = wer_metric.compute(
        predictions=predictions,
        references=references
    )
    print(f"\n===== CODESWITCH WER: {codeswitch_wer:.4f} =====")
else:
    codeswitch_wer = None
    print("\n===== CODESWITCH WER: could not compute =====")


# 패턴별 WER 계산
pattern_summary = {}

for pattern in sorted(set(item["pattern"] for item in all_results)):
    pattern_refs = [
        item["normalized_reference"]
        for item in all_results
        if item["pattern"] == pattern
    ]

    pattern_preds = [
        item["normalized_prediction"]
        for item in all_results
        if item["pattern"] == pattern
    ]

    if len(pattern_refs) > 0:
        pattern_wer = wer_metric.compute(
            predictions=pattern_preds,
            references=pattern_refs
        )
    else:
        pattern_wer = None

    pattern_summary[pattern] = {
        "wer": pattern_wer,
        "num_samples": len(pattern_refs),
    }

    if pattern_wer is not None:
        print(f"{pattern} WER: {pattern_wer:.4f}")
    else:
        print(f"{pattern} WER: could not compute")


# 결과 저장
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

final_output = {
    "model": BASE_MODEL,
    "type": "whisper_lora_codeswitch",
    "codeswitch_wer": codeswitch_wer,
    "num_total_samples": len(references),
    "pattern_summary": pattern_summary,
    "results": all_results,
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print(f"Saved code-switching results to {OUTPUT_JSON}")
