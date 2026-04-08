from datasets import load_dataset

# Common Voice 버전과 언어 코드
CV_VERSION = "11.0"
KO_LANG = "ko"
RU_LANG = "ru"

print("Loading Korean Common Voice...")
ko_ds = load_dataset(
    "mozilla-foundation/common_voice_" + CV_VERSION,
    KO_LANG,
    split="train",
    trust_remote_code=True
)

print("Loading Russian Common Voice...")
ru_ds = load_dataset(
    "mozilla-foundation/common_voice_" + CV_VERSION,
    RU_LANG,
    split="train",
    trust_remote_code=True
)

print("\n=== Dataset Loaded Successfully ===")
print("Korean train size:", len(ko_ds))
print("Russian train size:", len(ru_ds))

print("\n=== Korean Sample Keys ===")
print(ko_ds.column_names)

print("\n=== Russian Sample Keys ===")
print(ru_ds.column_names)