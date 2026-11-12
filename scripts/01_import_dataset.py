from pathlib import Path

import pandas as pd
from pymongo import MongoClient, ASCENDING

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "online_shoppers_intention.csv"

MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "smartcart"
COLLECTION_NAME = "shopping_sessions"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if "Revenue" not in df.columns:
        raise ValueError("Dataset must contain the Revenue target column.")

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values. Inspect before importing.")

    records = df.to_dict(orient="records")

    for record in records:
        record["Revenue"] = bool(record["Revenue"])

    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000
    )

    try:
        client.admin.command("ping")

        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        # Clear previous imported sessions to avoid duplicates on reruns.
        collection.delete_many({})

        if records:
            collection.insert_many(records, ordered=False)

        # Create indexes to speed up common queries.
        collection.create_index([("Revenue", ASCENDING)])
        collection.create_index([
            ("Month", ASCENDING),
            ("Revenue", ASCENDING)
        ])
        collection.create_index([("VisitorType", ASCENDING)])

        imported_count = collection.count_documents({})

        print("Import completed successfully.")
        print(f"Source rows: {len(df)}")
        print(f"MongoDB documents: {imported_count}")
        print(f"Database: {DATABASE_NAME}")
        print(f"Collection: {COLLECTION_NAME}")

        print("\nPurchase distribution:")

        for result in collection.aggregate([
            {
                "$group": {
                    "_id": "$Revenue",
                    "sessions": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]):
            print(
                f"Revenue={result['_id']}: "
                f"{result['sessions']} sessions"
            )

        print("\nSample document:")
        print(collection.find_one({}, {"_id": 0}))

    finally:
        client.close()


if __name__ == "__main__":
    main()