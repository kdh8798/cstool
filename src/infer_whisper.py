import os
import librosa
import torch
from tqdm import tqdm

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "whisper_lora_manual", "final")

# 테스트할 오디오 파일 경로
AUDIO_PATH = os.path.join(PROJECT_ROOT, "data", "samples", "test.mp3")


def load_audio(audio_path: str, target_sr: int = 16000):
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio, sr


def main():
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    if not os.path.exists(AUDIO_PATH):
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)

    print("Loading base model...")
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.to(device)
    model.eval()

    print("Loading audio...")
    audio, sr = load_audio(AUDIO_PATH, target_sr=16000)

    print("Preprocessing...")
    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(device)
    
    """
    한국어 테스트
                input_features=input_features,
                max_length=128,
                language="ko",
                task="transcribe"
    
    러시아어 테스트
                input_features=input_features,
                max_length=128,
                language="ru",
                task="transcribe"

    기본
                input_features=input_features,
                max_length=128
    """

    print("Running inference...")
    for _ in tqdm(range(1), desc="Inference progress"):
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features=input_features,
                max_length=128,
                language="ko",
                task="transcribe"
            )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    print("\n=== TRANSCRIPTION RESULT ===")
    print(transcription)


if __name__ == "__main__":
    main()
