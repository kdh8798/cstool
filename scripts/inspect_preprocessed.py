from datasets import load_from_disk

ko_path = "data/processed/commonvoice_ko"
ru_path = "data/processed/commonvoice_ru"

ko_ds = load_from_disk(ko_path)
ru_ds = load_from_disk(ru_path)

print("=== Korean Preprocessed Dataset ===")
print(ko_ds)
print("KO sample keys:", ko_ds.column_names)
print("KO first sample:")
print("input_features length:", len(ko_ds[0]["input_features"]))
print("labels length:", len(ko_ds[0]["labels"]))
print("normalized_text:", ko_ds[0]["normalized_text"])

print("\n=== Russian Preprocessed Dataset ===")
print(ru_ds)
print("RU sample keys:", ru_ds.column_names)
print("RU first sample:")
print("input_features length:", len(ru_ds[0]["input_features"]))
print("labels length:", len(ru_ds[0]["labels"]))
print("normalized_text:", ru_ds[0]["normalized_text"])