from datasets import load_dataset, Audio

CV_VERSION = "11.0"

ko_ds = load_dataset(
    "mozilla-foundation/common_voice_" + CV_VERSION,
    "ko",
    split="train[:3]",
    trust_remote_code=True
)

ru_ds = load_dataset(
    "mozilla-foundation/common_voice_" + CV_VERSION,
    "ru",
    split="train[:3]",
    trust_remote_code=True
)

# 오디오 컬럼을 16kHz로 디코딩
ko_ds = ko_ds.cast_column("audio", Audio(sampling_rate=16000))
ru_ds = ru_ds.cast_column("audio", Audio(sampling_rate=16000))

print("=== Korean Audio Samples ===")
for i in range(len(ko_ds)):
    sample = ko_ds[i]
    audio = sample["audio"]
    print(f"\n[KO Sample {i}]")
    print("Sentence:", sample["sentence"])
    print("Sampling Rate:", audio["sampling_rate"])
    print("Num Samples:", len(audio["array"]))
    print("Duration (sec):", len(audio["array"]) / audio["sampling_rate"])

print("\n=== Russian Audio Samples ===")
for i in range(len(ru_ds)):
    sample = ru_ds[i]
    audio = sample["audio"]
    print(f"\n[RU Sample {i}]")
    print("Sentence:", sample["sentence"])
    print("Sampling Rate:", audio["sampling_rate"])
    print("Num Samples:", len(audio["array"]))
    print("Duration (sec):", len(audio["array"]) / audio["sampling_rate"])