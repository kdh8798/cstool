import torch
from transformers import pipeline

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device
)

print("Whisper pipeline loaded successfully.")