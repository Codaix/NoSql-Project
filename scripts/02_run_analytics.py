from pymongo import MongoClient


MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "smartcart"
COLLECTION_NAME = "shopping_sessions"


def main():
    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000
    )

    try:
        client.admin.command("ping")
        collection = client[DATABASE_NAME][COLLECTION_NAME]

        print("\n=== OVERALL PURCHASE CONVERSION ===")
        overall = list(collection.aggregate([
            {
                "$group": {
                    "_id": None,
                    "totalSessions": {"$sum": 1},
                    "purchases": {
                        "$sum": {"$cond": ["$Revenue", 1, 0]}
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "totalSessions": 1,
                    "purchases": 1,
                    "conversionRate": {
                        "$round": [
                            {
                                "$multiply": [
                                    {"$divide": ["$purchases", "$totalSessions"]},
                                    100
                                ]
                            },
                            2
                        ]
                    }
                }
            }
        ]))
        print(overall)

        print("\n=== CONVERSION BY VISITOR TYPE ===")
        visitor_results = collection.aggregate([
            {
                "$group": {
                    "_id": "$VisitorType",
                    "totalSessions": {"$sum": 1},
                    "purchases": {
                        "$sum": {"$cond": ["$Revenue", 1, 0]}
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "visitorType": "$_id",
                    "totalSessions": 1,
                    "purchases": 1,
                    "conversionRate": {
                        "$round": [
                            {
                                "$multiply": [
                                    {"$divide": ["$purchases", "$totalSessions"]},
                                    100
                                ]
                            },
                            2
                        ]
                    }
                }
            },
            {"$sort": {"conversionRate": -1}}
        ])
        for row in visitor_results:
            print(row)

        print("\n=== CONVERSION BY WEEKEND ===")
        weekend_results = collection.aggregate([
            {
                "$group": {
                    "_id": "$Weekend",
                    "totalSessions": {"$sum": 1},
                    "purchases": {
                        "$sum": {"$cond": ["$Revenue", 1, 0]}
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "period": {
                        "$cond": ["$_id", "Weekend", "Weekday"]
                    },
                    "totalSessions": 1,
                    "purchases": 1,
                    "conversionRate": {
                        "$round": [
                            {
                                "$multiply": [
                                    {"$divide": ["$purchases", "$totalSessions"]},
                                    100
                                ]
                            },
                            2
                        ]
                    }
                }
            }
        ])
        for row in weekend_results:
            print(row)

        print("\n=== BROWSING BEHAVIOR BY PURCHASE OUTCOME ===")
        browsing_results = collection.aggregate([
            {
                "$group": {
                    "_id": "$Revenue",
                    "sessions": {"$sum": 1},
                    "avgProductPages": {"$avg": "$ProductRelated"},
                    "avgProductDuration": {"$avg": "$ProductRelated_Duration"},
                    "avgPageValues": {"$avg": "$PageValues"}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "purchaseMade": "$_id",
                    "sessions": 1,
                    "avgProductPages": {"$round": ["$avgProductPages", 2]},
                    "avgProductDuration": {"$round": ["$avgProductDuration", 2]},
                    "avgPageValues": {"$round": ["$avgPageValues", 2]}
                }
            }
        ])
        for row in browsing_results:
            print(row)

    finally:
        client.close()


if __name__ == "__main__":
    main()