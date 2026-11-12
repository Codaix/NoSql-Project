from pathlib import Path

import pandas as pd
from pymongo import MongoClient


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_PATH = OUTPUT_DIR / "ml_dataset.csv"

MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "smartcart"
COLLECTION_NAME = "shopping_sessions"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000
    )

    try:
        client.admin.command("ping")

        collection = client[DATABASE_NAME][COLLECTION_NAME]

        cursor = collection.find({}, {"_id": 0})
        df = pd.DataFrame(list(cursor))

        if df.empty:
            raise ValueError("MongoDB collection contains no documents.")

        if "Revenue" not in df.columns:
            raise ValueError("Target column 'Revenue' is missing.")

        df["Revenue"] = df["Revenue"].astype(bool)

        df.to_csv(OUTPUT_PATH, index=False)

        print("ML dataset exported successfully.")
        print(f"Output: {OUTPUT_PATH}")
        print(f"Rows: {len(df)}")
        print(f"Features including target: {len(df.columns)}")
        print(f"Missing values: {int(df.isna().sum().sum())}")
        print("\nTarget distribution:")
        print(df["Revenue"].value_counts().sort_index())

        print("\nColumns:")
        print(list(df.columns))

    finally:
        client.close()


if __name__ == "__main__":
    main()