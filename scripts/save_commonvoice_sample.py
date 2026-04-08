import soundfile as sf
from datasets import load_dataset, Audio

CV_VERSION = "11.0"

ko_ds = load_dataset(
    "mozilla-foundation/common_voice_" + CV_VERSION,
    "ko",
    split="train[:1]",
    trust_remote_code=True
)

ko_ds = ko_ds.cast_column("audio", Audio(sampling_rate=16000))

sample = ko_ds[0]
audio = sample["audio"]

print("Sentence:", sample["sentence"])
print("Saving sample to ko_sample.wav ...")

sf.write("ko_sample.wav", audio["array"], audio["sampling_rate"])
print("Saved successfully.")