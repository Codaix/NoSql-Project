import os
import json
import joblib
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
database_name = os.getenv("MONGODB_DATABASE", "smartcart")
collection_name = os.getenv("MONGODB_COLLECTION", "shopping_sessions")

print("Checking MongoDB connection...", flush=True)

client = MongoClient(uri, serverSelectionTimeoutMS=5000)

try:
    client.admin.command("ping")
    print("MongoDB connection: OK")

    collection = client[database_name][collection_name]
    print("Collection documents:", collection.count_documents({}))
finally:
    client.close()

model_path = "ml/saved_models/random_forest_tuned.joblib"
threshold_path = "ml/saved_models/threshold.json"

print("\nLoading saved model...")
model = joblib.load(model_path)
print("Saved model: OK")

with open(threshold_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print("Selected threshold:", metadata["threshold"])
print("Features:", len(metadata["features"]))
print("\nAll checks completed.")
