import librosa
from typing import Dict, Any
from transformers import WhisperProcessor

def prepare_dataset(sample: Dict[str, Any], processor: WhisperProcessor) -> Dict[str, Any]:
    audio_path = sample["audio"]
    text = sample["sentence"]

    # 직접 wav 로드 (torchcodec 안씀)
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)

    input_features = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate
    ).input_features[0]

    labels = processor.tokenizer(text).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
        "normalized_text": text
    }
