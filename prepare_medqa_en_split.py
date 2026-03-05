from datasets import DatasetDict, load_dataset

from config import OUTPUT_DIR, CACHE_DIR


def main() -> None:
    # MedQA requires trusting remote code to load the dataset script.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "bigbio/med_qa",
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
    )

    train_ds = ds["train"]
    validation_ds = ds["validation"]
    test_ds = ds["test"]

    total = len(train_ds) + len(validation_ds) + len(test_ds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict(
        {"train": train_ds, "validation": validation_ds, "test": test_ds}
    )
    dataset_dict.save_to_disk(str(OUTPUT_DIR / "hf_dataset"))

    train_ds.to_json(str(OUTPUT_DIR / "train.jsonl"))
    validation_ds.to_json(str(OUTPUT_DIR / "validation.jsonl"))
    test_ds.to_json(str(OUTPUT_DIR / "test.jsonl"))

    print(f"Total English questions: {total}")
    print(f"Train size: {len(train_ds)}")
    print(f"Validation size: {len(validation_ds)}")
    print(f"Test size: {len(test_ds)}")
    print(f"Saved files under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
