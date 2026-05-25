import os
import re
import json
import torch
import evaluate
from tqdm import tqdm
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
import warnings

from transformers.utils import logging as hf_logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# 터미널을 최대한 조용하게 zzz...
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# 1. 설정
BASE_MODEL = "openai/whisper-small"   # 학습에 사용한 base model로 변경

BASE_DIR = Path(__file__).resolve().parent.parent

# LORA_PATH = BASE_DIR / "outputs" / "whisper_lora_manual" / "final" # 네 LoRA adapter 경로로 변경
LORA_PATH = BASE_DIR / "outputs" / "whisper_lora_improved" / "final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = BASE_DIR / "data" / "raw"
LOCAL_DATASETS = {
    "ko": DATA_ROOT / "cv_ko",
    "ru": DATA_ROOT / "cv_ru",
}

# 평가 결과 저장 경로
OUTPUT_JSON = BASE_DIR / "results" / "wer_results.json"

# 로컬 Common Voice용 로더
def load_local_commonvoice(lang_dir: Path, split_name: str = "test", max_samples: int = None):
    """
    lang_dir 예시:
      data/raw/cv_ko
      data/raw/cv_ru

    기대 구조:
      lang_dir/test.tsv 또는 validated.tsv
      lang_dir/clips/*.mp3
    """
    split_path = lang_dir / f"{split_name}.tsv"

    if split_path.exists():
        tsv_path = split_path
    else:
        # test.tsv가 없으면 validated.tsv 사용
        fallback = lang_dir / "validated.tsv"
        if fallback.exists():
            tsv_path = fallback
        else:
            raise FileNotFoundError(
                f"'{lang_dir}' 안에서 {split_name}.tsv 또는 validated.tsv 를 찾지 못함"
            )

    df = pd.read_csv(tsv_path, sep="\t")

    # Common Voice는 보통 'path', 'sentence' 컬럼이 있음
    required_cols = ["path", "sentence"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{tsv_path} 에 '{col}' 컬럼이 없음. 현재 컬럼: {list(df.columns)}")

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

# mp3 파일 읽기
def load_audio_file(audio_path: Path, target_sr: int = 16000):
    audio_array, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)

    # float32로 통일
    audio_array = np.asarray(audio_array, dtype=np.float32)

    return audio_array, sr

# 2. 텍스트 정규화 함수
def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    # 필요시 구두점 제거
    text = re.sub(r"[^\w\s가-힣а-яё]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# 3. 모델 / 프로세서 로드
print("Loading processor...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.to(DEVICE)
model.eval()

model.generation_config.max_length = None

# generation 설정
# model.config.forced_decoder_ids = None
# model.config.suppress_tokens = []

# 4. WER metric 준비
wer_metric = evaluate.load("wer")

# 5. 추론 함수
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

# 6. 언어별 평가
all_results = []
overall_references = []
overall_predictions = []

for lang_name, lang_dir in LOCAL_DATASETS.items():
    print(f"\n===== Evaluating language: {lang_name} =====")

    samples = load_local_commonvoice(
        lang_dir=lang_dir,
        split_name="test",      # test.tsv 있으면 사용
        max_samples=None        # 빠른 테스트는 20 같은 숫자 넣기
    )

    print(f"Loaded {len(samples)} samples from {lang_dir}")

    lang_references = []
    lang_predictions = []

    for idx, sample in enumerate(tqdm(samples, desc=f"Evaluating {lang_name}")):
        try:
            audio_array, sampling_rate = load_audio_file(sample["audio_path"])
            reference = sample["sentence"]

            prediction = transcribe_audio(audio_array, sampling_rate, lang_code=lang_name)

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
                "normalized_prediction": norm_pred
            })

        except Exception as e:
            print(f"[ERROR] {lang_name}_{idx} | {sample['audio_path'].name} | {type(e).__name__}: {e}")

    if len(lang_references) > 0:
        lang_wer = wer_metric.compute(
            predictions=lang_predictions,
            references=lang_references
        )
        print(f"{lang_name.upper()} WER: {lang_wer:.4f}")
    else:
        lang_wer = None
        print(f"{lang_name.upper()} WER: could not compute")

    all_results.append({
        "summary": True,
        "lang": lang_name,
        "wer": lang_wer,
        "num_samples": len(lang_references)
    })

# 7. 전체 WER 계산
if len(overall_references) > 0:
    overall_wer = wer_metric.compute(
        predictions=overall_predictions,
        references=overall_references
    )
    print(f"\n===== OVERALL WER: {overall_wer:.4f} =====")
else:
    overall_wer = None
    print("\n===== OVERALL WER: could not compute =====")

# 8. 결과 저장
final_output = {
    "overall_wer": overall_wer,
    "num_total_samples": len(overall_references),
    "results": all_results
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print(f"Saved results to {OUTPUT_JSON}")
