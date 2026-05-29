"""
프로젝트 루트에서 실행:
cd C:\Github\cstool
python src\run_pipeline.py data\samples\test.mp3

코드스위칭 음성 (자동 감지):
python src\run_pipeline.py data\samples\test.mp3 --language auto

한국어 테스트:
python src\run_pipeline.py data\samples\test.mp3 --language ko

러시아어 테스트:
python src\run_pipeline.py data\samples\test.mp3 --language ru

역할:
1) audio 파일 경로 입력
2) Whisper + LoRA 추론
3) 텍스트 출력
4) feedback_generator.generate_feedback() 호출
5) 피드백 출력
"""

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import argparse
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging
from peft import PeftModel

from src.feedback_generator import generate_feedback, generate_feedback_from_analysis
from src.llm_postprocessor import correct_stt_text, postprocess_feedback, analyze_codeswitch_text

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


# 경로 / 모델 설정
BASE_MODEL = "openai/whisper-small"

BASE_DIR = Path(__file__).resolve().parent.parent

# 최종 모델이 있으면 이 경로 사용
FINAL_LORA_PATH = BASE_DIR / "outputs" / "whisper_lora_improved" / "final"

# final 모델 학습 전이면 기존 manual 모델 사용
MANUAL_LORA_PATH = BASE_DIR / "outputs" / "whisper_lora_manual" / "final"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000


# 사용할 LoRA 경로 선택
def get_lora_path():
    if FINAL_LORA_PATH.exists():
        return FINAL_LORA_PATH

    if MANUAL_LORA_PATH.exists():
        print("[WARN] whisper_lora_final을 찾지 못해 whisper_lora_manual/final을 사용합니다.")
        return MANUAL_LORA_PATH

    raise FileNotFoundError(
        "사용 가능한 LoRA 모델 폴더를 찾지 못했습니다.\n"
        f"확인 경로 1: {FINAL_LORA_PATH}\n"
        f"확인 경로 2: {MANUAL_LORA_PATH}"
    )


# 오디오 로딩
def load_audio(audio_path: Path):
    if not audio_path.exists():
        raise FileNotFoundError(f"오디오 파일을 찾지 못했습니다: {audio_path}")

    audio, sr = librosa.load(
        str(audio_path),
        sr=TARGET_SR,
        mono=True
    )

    audio = np.asarray(audio, dtype=np.float32)
    return audio, sr


# 모델 로드
def load_asr_model(lora_path: Path):
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(lora_path)

    print("Loading base model...")
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    model = model.to(DEVICE)
    model.eval()

    # max_new_tokens / max_length 중복 경고 방지
    model.generation_config.max_length = None

    return processor, model


# Whisper 추론
@torch.no_grad()
def transcribe_audio(processor, model, audio, sr, language=None, use_prompt=False):
    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(DEVICE)

    generate_kwargs = {
        "max_new_tokens": 64,
        "task": "transcribe",
        "num_beams": 5,
        "do_sample": False,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
        "return_dict_in_generate": True,
    }

    if language is not None and language.lower() != "auto":
        generate_kwargs["language"] = language

    """
    if use_prompt:
        prompt_ids = processor.get_prompt_ids(
            "дробь знаменатель числитель что это число что как где почему сколько я хочу"
        )

        generate_kwargs["prompt_ids"] = torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=DEVICE
        )
    """

    predicted_ids = model.generate(
        input_features=input_features,
        **generate_kwargs
    )

    transcription = processor.batch_decode(
        predicted_ids.sequences,
        skip_special_tokens=True
    )[0]

    return transcription.strip()


# STT 다중 후보 생성
def transcribe_audio_candidates(processor, model, audio, sr):
    candidates = {}

    configs = [
        ("auto", None, False),
        ("ko", "ko", False),
        ("ru", "ru", False),
    ]

    for name, lang, use_prompt in configs:
        try:
            result = transcribe_audio(
                processor=processor,
                model=model,
                audio=audio,
                sr=sr,
                language=lang,
                use_prompt=use_prompt,
            )
            candidates[name] = result
        except Exception as e:
            candidates[name] = f"[ERROR] {type(e).__name__}: {e}"

    return candidates


# 대표값 선택
def score_asr_candidate(text: str) -> int:
    if not text or text.startswith("[ERROR]"):
        return -999

    score = 0

    # 너무 짧으면 감점
    if len(text.strip()) < 3:
        score -= 5

    # 반복 문장 감점
    words = text.split()
    if len(words) >= 4:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:
            score -= 10

    # 한글 포함 가점
    if any("가" <= ch <= "힣" for ch in text):
        score += 3

    # 키릴 문자 포함 가점
    if any("\u0400" <= ch <= "\u04FF" for ch in text):
        score += 4

    # 의미 있는 질문 단어 가점
    for marker in ["뭐", "어디", "왜", "어떻게", "맞아", "뜻", "숫자", "분모", "분자"]:
        if marker in text:
            score += 2

    # 이상한 반복 감점
    if text.count("이거") >= 5:
        score -= 20

    return score


def select_best_candidate(candidates: dict) -> str:
    if not candidates:
        return ""

    scored = [
        (name, text, score_asr_candidate(str(text)))
        for name, text in candidates.items()
    ]

    scored.sort(key=lambda x: x[2], reverse=True)

    print("\n[ASR CANDIDATE SCORES]")
    for name, text, score in scored:
        print(f"{name}: {score} | {text}")

    return scored[0][1]


def get_scored_candidates(candidates: dict) -> list[dict]:
    scored = []

    for name, text in candidates.items():
        scored.append({
            "name": name,
            "text": str(text),
            "score": score_asr_candidate(str(text)),
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


# 전체 파이프라인 실행
def run_pipeline(audio_path: Path, language=None):
    lora_path = get_lora_path()

    processor, model = load_asr_model(lora_path)

    print("Loading audio...")
    audio, sr = load_audio(audio_path)

    print("Running inference...")
    scored_candidates = []

    if language is None or language.lower() == "auto":
        candidates = transcribe_audio_candidates(
            processor=processor,
            model=model,
            audio=audio,
            sr=sr,
        )

        transcription = select_best_candidate(candidates)
        scored_candidates = get_scored_candidates(candidates)
    else:
        candidates = {}
        transcription = transcribe_audio(
            processor=processor,
            model=model,
            audio=audio,
            sr=sr,
            language=language,
        )
    
    corrected_transcription = correct_stt_text(
        transcription,
        candidates=candidates,
        scored_candidates=scored_candidates,
    )

    analysis = analyze_codeswitch_text(
        stt_text=corrected_transcription,
        candidates=candidates,
    )

    final_transcription = analysis.get("corrected_text", corrected_transcription)

    rule_feedback = generate_feedback_from_analysis(analysis)
    feedback = postprocess_feedback(final_transcription, rule_feedback)

    print("\n========== PIPELINE RESULT ==========")
    print(f"[AUDIO]")
    print(audio_path)

    if candidates:
        print("\n[ASR CANDIDATES]")
        for lang, text in candidates.items():
            print(f"{lang}: {text}")

    print("\n[RAW TRANSCRIPTION]")
    print(transcription)

    print("\n[CORRECTED TRANSCRIPTION]")
    print(corrected_transcription)

    print("\n[ANALYSIS]")
    print(analysis)

    print("\n[FINAL TRANSCRIPTION]")
    print(final_transcription)

    print("\n[FEEDBACK]")
    print(feedback)

    print("=====================================")

    return {
        "audio_path": str(audio_path),
        "raw_transcription": transcription,
        "asr_candidates": candidates,
        "transcription": final_transcription,
        "corrected_transcription": corrected_transcription,
        "analysis": analysis,
        "rule_feedback": rule_feedback,
        "feedback": feedback,
        "lora_path": str(lora_path),
        "scored_candidates": scored_candidates,
    }


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio → Whisper LoRA ASR → Feedback pipeline"
    )

    parser.add_argument(
        "audio",
        type=str,
        help="입력 오디오 파일 경로 (예: data/samples/test.mp3)"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="언어 지정: ko, ru, auto. 코드스위칭은 auto 추천"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    audio_path = Path(args.audio)

    if not audio_path.is_absolute():
        audio_path = BASE_DIR / audio_path

    run_pipeline(
        audio_path=audio_path,
        language=args.language
    )


if __name__ == "__main__":
    main()
