import torch
import transformers
import datasets
import peft
import librosa
import soundfile as sf
print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("PEFT:", peft.__version__)
print("Librosa:", librosa.__version__)
print("SoundFile:", sf.__libsndfile_version__)
print("Environment setup complete.")