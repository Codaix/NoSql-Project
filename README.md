# 🛒 SmartCart — E-Commerce Purchase Intention Analytics

**A MongoDB-based analytics and machine learning application for understanding and predicting online shopping purchase behaviour.**

SmartCart analyses e-commerce browsing sessions using MongoDB aggregation pipelines and applies a tuned Random Forest classification model to estimate whether a visitor is likely to make a purchase.

The project combines NoSQL database operations, data analytics, machine learning, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

The objective of SmartCart is to explore online shopping behaviour and identify patterns associated with purchase decisions.

The application provides three main capabilities:

- **Business Analytics:** Analyse shopping sessions, conversion rates, visitor types, and browsing behaviour.
- **Purchase Prediction:** Estimate the probability of a purchase from user-provided session characteristics.
- **Model Performance:** Examine and compare classification metrics before and after decision-threshold tuning.

### Dataset

The project uses the **Online Shoppers Purchasing Intention Dataset**.

| Attribute | Value |
|---|---:|
| Total records | 12,330 |
| Features | 18 |
| Purchase sessions | 1,908 |
| Non-purchase sessions | 10,422 |
| Purchase conversion rate | 15.47% |

The target variable is `Revenue`, which indicates whether a shopping session resulted in a purchase.

---

## 🧭 Repository Navigation

Start with `app.py` to understand the application, then explore the folders below to understand how the database, analytics, and machine learning components are organised.

| File / Folder | Purpose |
|---|---|
| [`app.py`](./app.py) | Main Streamlit application and dashboard entry point |
| [`data/`](./data/) | Data-related project resources |
| [`database/`](./database/) | MongoDB-related resources and database operations |
| [`ml/`](./ml/) | Machine learning pipeline, training, evaluation, and model-related code |
| [`reports/`](./reports/) | Generated reports and model evaluation results |
| [`scripts/`](./scripts/) | Supporting scripts for project execution and processing |
| [`services/`](./services/) | Supporting application service modules |
| [`online_shoppers_intention.csv`](./online_shoppers_intention.csv) | Original dataset used by the project |
| [`check_dashboard.py`](./check_dashboard.py) | Utility script to verify MongoDB connectivity and saved model artifacts |

### Important application files

The dashboard expects the following model and report artifacts:

```text
ml/
└── saved_models/
    ├── random_forest_tuned.joblib
    └── threshold.json

reports/
└── threshold_comparison.csv
```

These files are used by the dashboard to load the trained model, retrieve the selected classification threshold, and display model evaluation results.

---

## 🏗️ Application Architecture

```text
Online Shoppers Dataset
          |
          v
    Data Processing
          |
          v
       MongoDB
          |
          v
  PyMongo + Aggregation
          |
          v
   Streamlit Dashboard
          |
          +----------------------+
          |                      |
          v                      v
  Business Analytics      Purchase Prediction
                                 |
                                 v
                         Tuned Random Forest
                                 |
                                 v
                         Prediction Probability
                                 |
                                 v
                       Threshold-based Decision
```

### Technology Stack

| Technology | Role |
|---|---|
| Python | Core programming language |
| MongoDB | NoSQL document database |
| PyMongo | Python–MongoDB connectivity |
| Streamlit | Interactive dashboard |
| Pandas | Data manipulation |
| Plotly | Interactive visualisations |
| Scikit-learn | Machine learning and evaluation |
| Joblib | Saving and loading trained model artifacts |

---

## 📊 Dashboard Navigation

The Streamlit application contains three sections accessible through the sidebar.

### 1. Business Analytics

Provides an overview of online shopping activity, including:

- Total shopping sessions
- Purchase and non-purchase sessions
- Overall conversion rate
- Session distribution by visitor type
- Conversion rate by visitor type
- Weekday versus weekend conversion
- Browsing behaviour by purchase outcome

MongoDB aggregation pipelines are used to calculate these analytics directly from the collection.

### 2. Purchase Prediction

Allows users to enter characteristics of a shopping session, such as:

- Visitor type and month
- Administrative and informational page activity
- Product-related page views and duration
- Bounce rate and exit rate
- Browser, operating system, region, and traffic type
- Weekend and special-day information

The trained Random Forest model returns an estimated purchase probability and a classification.

### 3. Model Performance

Displays the model evaluation report and compares the default classification threshold with the tuned threshold.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Threshold tuning is used to adjust the classification decision according to the validation-selected threshold. It does not retrain the model or change its underlying probability ranking.

---

## ⚙️ Getting Started

### Prerequisites

Install the following before running the project:

- Python 3.10 or a compatible Python version
- MongoDB Community Server
- Git
- A Python environment manager (optional but recommended)

### 1. Clone the repository

```bash
git clone https://github.com/Codaix/NoSql-Project.git
cd NoSql-Project
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the packages required by the application:

```bash
pip install streamlit pymongo python-dotenv pandas plotly scikit-learn joblib
```

### 4. Configure MongoDB

Make sure your MongoDB server is running.

Create a `.env` file in the project root:

```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=smartcart
MONGODB_COLLECTION=shopping_sessions
```

The application uses the following default database and collection:

```text
Database:   smartcart
Collection: shopping_sessions
```

**Important:** The dashboard expects the MongoDB collection to contain the project dataset. Configure the database and import the dataset before launching the dashboard if the collection is empty.

Never commit `.env` or real database credentials to GitHub.

### 5. Run the application

From the project root:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

---

## 🤖 Machine Learning Approach

SmartCart uses a supervised binary classification approach to predict purchase intention.

**Model:** Tuned Random Forest Classifier

**Target:** `Revenue`

The model estimates the probability of a purchase and applies a selected decision threshold to classify the session.

### Why threshold tuning?

The default classification threshold of 0.5 may not be ideal when the dataset contains substantially more non-purchase sessions than purchase sessions.

Threshold tuning allows the classification decision to be adjusted to improve the balance between precision and recall.

The tuned threshold is stored in `ml/saved_models/threshold.json`.

---

## 🔍 Database Concepts Demonstrated

This project demonstrates practical MongoDB and NoSQL concepts, including:

- Database and collection creation
- Document insertion and retrieval
- BSON document structure
- Query filtering
- Index creation
- Compound indexing
- Schema validation
- Aggregation pipelines
- Grouping and sorting
- Conditional aggregation
- Data extraction for machine learning

---

## 🧪 Verification

A utility script is included to check the MongoDB connection, document count, and saved model artifacts.

Run it from the project root:

```bash
python check_dashboard.py
```

---

## 📚 Academic Context

**Course:** CSE494 — Intelligent NoSQL Databases  
**Academic Session:** 2026–2027

This project demonstrates the integration of MongoDB-based data management, analytical queries, and machine learning in an interactive application.

---

## 👨‍💻 Author

**Pranav Sharma**

GitHub: [@Codaix](https://github.com/Codaix)

**Repository:** [NoSql-Project](https://github.com/Codaix/NoSql-Project)

---

*SmartCart — Turning online shopping session data into actionable insights.*
