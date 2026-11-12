
import os
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

MONGODB_URI = os.getenv(
    "MONGODB_URI", "mongodb://localhost:27017/"
)
DATABASE = os.getenv("MONGODB_DATABASE", "smartcart")
COLLECTION = os.getenv(
    "MONGODB_COLLECTION", "shopping_sessions"
)

MODEL_PATH = ROOT / "ml" / "saved_models" / "random_forest_tuned.joblib"
THRESHOLD_PATH = ROOT / "ml" / "saved_models" / "threshold.json"
REPORT_PATH = ROOT / "reports" / "threshold_comparison.csv"

st.set_page_config(
    page_title="SmartCart Analytics",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 SmartCart")
st.caption("E-Commerce Purchase Intention Analytics")

# --------------------------------------------------
# 2. Load saved model and threshold
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


@st.cache_data(ttl=30)
def load_analytics():
    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000,
    )
    try:
        client.admin.command("ping")
        collection = client[DATABASE][COLLECTION]

        total = collection.count_documents({})
        purchases = collection.count_documents({"Revenue": True})
        non_purchases = collection.count_documents({"Revenue": False})

        visitor_pipeline = [
            {
                "$group": {
                    "_id": "$VisitorType",
                    "sessions": {"$sum": 1},
                    "purchases": {
                        "$sum": {
                            "$cond": ["$Revenue", 1, 0]
                        }
                    },
                }
            },
            {"$sort": {"sessions": -1}},
        ]

        visitor_rows = list(collection.aggregate(visitor_pipeline))

        weekend_pipeline = [
            {
                "$group": {
                    "_id": "$Weekend",
                    "sessions": {"$sum": 1},
                    "purchases": {
                        "$sum": {
                            "$cond": ["$Revenue", 1, 0]
                        }
                    },
                }
            },
            {"$sort": {"_id": 1}},
        ]

        weekend_rows = list(collection.aggregate(weekend_pipeline))

        browsing_pipeline = [
            {
                "$group": {
                    "_id": "$Revenue",
                    "sessions": {"$sum": 1},
                    "avg_product_pages": {
                        "$avg": "$ProductRelated"
                    },
                    "avg_product_duration": {
                        "$avg": "$ProductRelated_Duration"
                    },
                    "avg_page_values": {"$avg": "$PageValues"},
                }
            },
            {"$sort": {"_id": 1}},
        ]

        browsing_rows = list(collection.aggregate(browsing_pipeline))

        return (
            total, purchases, non_purchases,
            visitor_rows, weekend_rows, browsing_rows,
        )
    finally:
        client.close()


try:
    model, metadata = load_model()
    (
        total, purchases, non_purchases,
        visitor_rows, weekend_rows, browsing_rows,
    ) = load_analytics()
except Exception as e:
    st.error(f"Unable to load the model or MongoDB data: {e}")
    st.stop()

# --------------------------------------------------
# 3. Sidebar
# --------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select section",
    [
        "Business Analytics",
        "Purchase Prediction",
        "Model Performance",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Data source: MongoDB")
st.sidebar.caption("Model: Tuned Random Forest")
st.sidebar.caption(
    f"Decision threshold: {metadata['threshold']:.3f}"
)

# --------------------------------------------------
# 4. Business analytics
# --------------------------------------------------

if page == "Business Analytics":
    st.header("Business Analytics")

    conversion = purchases / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sessions", f"{total:,}")
    c2.metric("Purchase Sessions", f"{purchases:,}")
    c3.metric("Non-Purchase Sessions", f"{non_purchases:,}")
    c4.metric("Conversion Rate", f"{conversion:.2f}%")

    st.divider()

    visitor_df = pd.DataFrame(visitor_rows)
    visitor_df = visitor_df.rename(
        columns={"_id": "Visitor Type"}
    )
    visitor_df["Conversion Rate (%)"] = (
        visitor_df["purchases"] / visitor_df["sessions"] * 100
    )

    weekend_df = pd.DataFrame(weekend_rows)
    weekend_df["Day Type"] = weekend_df["_id"].map(
        {True: "Weekend", False: "Weekday"}
    )
    weekend_df["Conversion Rate (%)"] = (
        weekend_df["purchases"] / weekend_df["sessions"] * 100
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Sessions by Visitor Type")
        fig = px.bar(
            visitor_df,
            x="Visitor Type",
            y="sessions",
            color="Visitor Type",
            title="Session Volume",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Conversion by Visitor Type")
        fig = px.bar(
            visitor_df,
            x="Visitor Type",
            y="Conversion Rate (%)",
            color="Visitor Type",
            title="Purchase Conversion Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekday vs Weekend")
    fig = px.bar(
        weekend_df,
        x="Day Type",
        y="Conversion Rate (%)",
        color="Day Type",
        title="Conversion Rate by Day Type",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Visitor Type Summary")
    st.dataframe(
        visitor_df[
            ["Visitor Type", "sessions", "purchases", "Conversion Rate (%)"]
        ].round(2),
        use_container_width=True,
        hide_index=True,
    )

    browsing_df = pd.DataFrame(browsing_rows)
    browsing_df["Purchase Outcome"] = browsing_df["_id"].map(
        {True: "Purchase", False: "No Purchase"}
    )

    st.subheader("Browsing Behaviour by Purchase Outcome")
    st.dataframe(
        browsing_df[
            [
                "Purchase Outcome",
                "sessions",
                "avg_product_pages",
                "avg_product_duration",
                "avg_page_values",
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These are observed associations in session data, not evidence of causation. "
        "PageValues is a dataset feature, not actual sales revenue."
    )

# --------------------------------------------------
# 5. Purchase prediction
# --------------------------------------------------

elif page == "Purchase Prediction":
    st.header("Purchase Intention Predictor")
    st.write(
        "Enter the characteristics of a browsing session. "
        "The model estimates purchase likelihood."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            month = st.selectbox(
                "Month",
                [
                    "Feb", "Mar", "May", "June", "Jul",
                    "Aug", "Sep", "Oct", "Nov", "Dec",
                    "Jan", "Apr",
                ],
                index=4,
            )
            visitor_type = st.selectbox(
                "Visitor Type",
                [
                    "Returning_Visitor",
                    "New_Visitor",
                    "Other",
                ],
            )
            weekend = st.selectbox(
                "Weekend Session", [False, True],
                format_func=lambda x: "Yes" if x else "No",
            )
            special_day = st.slider(
                "Special Day Proximity", 0.0, 1.0, 0.0, 0.1
            )
            administrative = st.number_input(
                "Administrative Pages", 0, 100, 0
            )
            administrative_duration = st.number_input(
                "Administrative Duration", 0.0, value=0.0
            )
            informational = st.number_input(
                "Informational Pages", 0, 100, 0
            )
            informational_duration = st.number_input(
                "Informational Duration", 0.0, value=0.0
            )

        with col2:
            product_related = st.number_input(
                "Product-Related Pages", 0, 500, 10
            )
            product_duration = st.number_input(
                "Product-Related Duration", 0.0, value=300.0
            )
            bounce_rate = st.number_input(
                "Bounce Rate", 0.0, 1.0, 0.02, 0.01
            )
            exit_rate = st.number_input(
                "Exit Rate", 0.0, 1.0, 0.04, 0.01
            )
            operating_systems = st.number_input(
                "Operating System Code", 1, 10, 2
            )
            browser = st.number_input(
                "Browser Code", 1, 20, 2
            )
            region = st.number_input(
                "Region Code", 1, 10, 1
            )
            traffic_type = st.number_input(
                "Traffic Type Code", 1, 30, 2
            )

        submitted = st.form_submit_button(
            "Predict Purchase Intention",
            use_container_width=True,
        )

    if submitted:
        input_data = pd.DataFrame([{
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_duration,
            "BounceRates": bounce_rate,
            "ExitRates": exit_rate,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend,
        }])

        probability = model.predict_proba(input_data)[0, 1]
        threshold = metadata["threshold"]
        prediction = probability >= threshold

        st.divider()
        st.subheader("Prediction Result")

        m1, m2 = st.columns(2)
        m1.metric(
            "Estimated Purchase Probability",
            f"{probability:.1%}",
        )
        m2.metric(
            "Classification",
            "Purchase Likely" if prediction else "Purchase Unlikely",
        )

        st.progress(float(probability))

        st.caption(
            f"Classification uses the validation-selected threshold "
            f"of {threshold:.3f}. This is a model estimate, not a guarantee."
        )

# --------------------------------------------------
# 6. Model performance
# --------------------------------------------------

elif page == "Model Performance":
    st.header("Model Performance")
    st.write(
        "Metrics are calculated on the held-out test set "
        "and loaded from the generated report."
    )

    if not REPORT_PATH.exists():
        st.warning(
            "The threshold comparison report was not found. "
            "Run ml/03_tune_threshold.py first."
        )
    else:
        report_df = pd.read_csv(REPORT_PATH)
        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True,
        )

        metric_columns = [
            c for c in [
                "accuracy", "precision", "recall", "f1", "roc_auc"
            ] if c in report_df.columns
        ]

        if metric_columns:
            melted = report_df.melt(
                id_vars=["configuration"],
                value_vars=metric_columns,
                var_name="Metric",
                value_name="Score",
            )
            fig = px.bar(
                melted,
                x="Metric",
                y="Score",
                color="configuration",
                barmode="group",
                title="Default vs Tuned Threshold",
                range_y=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Threshold tuning changes the classification decision, "
            "not the underlying probability ranking. Therefore ROC-AUC "
            "should remain the same for both thresholds."
        )

st.divider()
st.caption("SmartCart | MongoDB + PyMongo + Scikit-learn + Streamlit")
