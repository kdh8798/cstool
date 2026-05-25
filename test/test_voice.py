import torch
from transformers import pipeline

audio_path = "" \
""

device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=device
)

result = pipe(audio_path)

print("=== RESULT ===")
print(result["text"])