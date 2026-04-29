"""Prepare and save processed train/validation/test datasets."""

from __future__ import annotations

from src.data import feature_columns, load_modeling_dataset, save_processed_datasets


def main() -> None:
    paths = save_processed_datasets()
    dataset = load_modeling_dataset()
    features = feature_columns(dataset)

    print("Processed datasets saved:")
    for name, path in paths.items():
        print(f"- {name}: {path}")

    print("\nDataset summary:")
    print(f"- rows: {len(dataset):,}")
    print(f"- features: {len(features):,}")
    print("- split counts:")
    print(dataset["split"].value_counts().sort_index().to_string())
    print("- target mean by split:")
    print(dataset.groupby("split")["target"].mean().sort_index().round(4).to_string())


if __name__ == "__main__":
    main()
