import soundfile as sf
from datasets import load_dataset, Audio

CV_VERSION = "11.0"

def check_dataset(lang_code: str, lang_name: str):
    print(f"\n=== Loading {lang_name} Common Voice ===")

    ds = load_dataset(
        "mozilla-foundation/common_voice_" + CV_VERSION,
        lang_code,
        split="train[:2]",
        trust_remote_code=True
    )

    print(f"{lang_name} columns:", ds.column_names)

    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    for i in range(len(ds)):
        sample = ds[i]
        audio = sample["audio"]

        print(f"\n[{lang_name} Sample {i}]")
        print("Sentence:", sample["sentence"])
        print("Sampling Rate:", audio["sampling_rate"])
        print("Num Samples:", len(audio["array"]))
        print("Duration (sec):", len(audio["array"]) / audio["sampling_rate"])

        out_name = f"{lang_code}_sample_{i}.wav"
        sf.write(out_name, audio["array"], audio["sampling_rate"])
        print("Saved:", out_name)

check_dataset("ko", "Korean")
check_dataset("ru", "Russian")