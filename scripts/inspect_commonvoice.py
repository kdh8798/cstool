from datasets import load_dataset

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

print("=== Korean Samples ===")
for i in range(len(ko_ds)):
    sample = ko_ds[i]
    print(f"\n[KO Sample {i}]")
    print("Sentence:", sample.get("sentence"))
    print("Path:", sample.get("path"))
    print("Client ID:", sample.get("client_id"))
    print("Up Votes:", sample.get("up_votes"))
    print("Down Votes:", sample.get("down_votes"))
    print("Age:", sample.get("age"))
    print("Gender:", sample.get("gender"))
    print("Accent:", sample.get("accent"))

print("\n=== Russian Samples ===")
for i in range(len(ru_ds)):
    sample = ru_ds[i]
    print(f"\n[RU Sample {i}]")
    print("Sentence:", sample.get("sentence"))
    print("Path:", sample.get("path"))
    print("Client ID:", sample.get("client_id"))
    print("Up Votes:", sample.get("up_votes"))
    print("Down Votes:", sample.get("down_votes"))
    print("Age:", sample.get("age"))
    print("Gender:", sample.get("gender"))
    print("Accent:", sample.get("accent"))