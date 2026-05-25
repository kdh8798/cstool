import os
import re
import librosa


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    # 한글 / 러시아어 / 영어 / 숫자 / 공백만 유지
    text = re.sub(r"[^0-9A-Za-z가-힣А-Яа-яЁё\s]", " ", text)

    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_sample(sample):
    audio_path = sample["audio"]
    text = sample["sentence"]

    if not os.path.exists(audio_path):
        return False

    if text is None or len(str(text).strip()) == 0:
        return False

    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
    except Exception as e:
        print(f"Failed to load audio: {audio_path} | {e}")
        return False

    return 1.0 <= duration <= 15.0
