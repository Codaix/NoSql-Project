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
| `app.py` | Main Streamlit application and dashboard entry point |
| `data/` | Data-related project resources |
| `database/` | MongoDB-related resources and database operations |
| `ml/` | Machine learning pipeline, training, evaluation, and model-related code |
| `pages/` | Additional dashboard pages |
| `reports/` | Generated reports and model evaluation results |
| `scripts/` | Supporting scripts for project execution and processing |
| `services/` | Supporting application service modules |
| `tests/` | Project testing resources |
| `online_shoppers_intention.csv` | Original dataset |
| `check_dashboard.py` | Utility script to verify MongoDB connectivity and saved model artifacts |

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

**Important:** Ensure these required artifacts are included in the GitHub repository. The dashboard cannot perform predictions if the trained model files are missing.

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
 Business Analytics     Purchase Prediction
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
| Joblib | Saving and loading model artifacts |

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

Threshold tuning adjusts the classification decision according to the validation-selected threshold. It does not retrain the model or change its underlying probability ranking.

---

# ⚙️ Getting Started — Run SmartCart on Your Own PC

Follow these instructions to download, configure, and run SmartCart locally.

You do not need to use the original developer's computer. You only need the project repository, its dependencies, MongoDB, and the required project files.

## Prerequisites

Install the following:

1. **Python 3.10 or a compatible version**
2. **MongoDB Community Server**
3. **MongoDB Shell (mongosh)**
4. **Git**
5. A terminal such as Windows PowerShell or Linux/macOS Terminal.

Useful installation links:

- Python: https://www.python.org/downloads/
- MongoDB: https://www.mongodb.com/try/download/community
- MongoDB Shell: https://www.mongodb.com/try/download/shell
- Git: https://git-scm.com/downloads/

## Step 1. Clone the GitHub repository

Open PowerShell or a terminal and execute:

```bash
git clone https://github.com/Codaix/NoSql-Project.git
```

Navigate into the downloaded project:

```bash
cd NoSql-Project
```

## Step 2. Create a virtual environment

A virtual environment keeps project dependencies separate from other Python installations.

**Windows PowerShell:**

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again.

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3. Install dependencies

With the virtual environment activated, install the required packages:

```bash
pip install streamlit pymongo python-dotenv pandas plotly scikit-learn joblib
```

If the repository contains a `requirements.txt` file, you can instead install the dependencies using:

```bash
pip install -r requirements.txt
```

## Step 4. Start MongoDB

Make sure the MongoDB Community Server is installed and running.

On Windows, MongoDB can usually be started through the Windows Services application if it was installed as a service.

You can verify the MongoDB Shell connection using:

```bash
mongosh
```

Inside the shell, execute:

```javascript
show dbs
```

Exit the shell:

```javascript
exit
```

## Step 5. Configure the database connection

Create a file named `.env` in the root directory of the project.

Add the following configuration:

```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=smartcart
MONGODB_COLLECTION=shopping_sessions
```

The application uses:

```text
Database:   smartcart
Collection: shopping_sessions
```

The `.env` file contains local configuration and should not be uploaded to GitHub if it contains private credentials.

## Step 6. Import the dataset into MongoDB

**This step is essential for a fresh installation.**

Cloning the GitHub repository does not automatically populate the new user's MongoDB database.

The original dataset is available in:

```text
online_shoppers_intention.csv
```

Run the database initialization or dataset import script provided in the project's `database/` or `scripts/` directory.

Make sure the import process:

- Connects to the `smartcart` database.
- Imports the dataset into `shopping_sessions`.
- Creates the required indexes.
- Applies the collection validation rules, where applicable.

Verify that the collection contains the expected records (12,330 documents).

If the database has already been initialized, avoid importing the dataset again unless the import script safely handles existing records.

## Step 7. Verify the project setup

Run the included verification script:

```bash
python check_dashboard.py
```

This checks MongoDB connectivity, document count, and saved model artifacts.

Resolve any missing database or model-file issues before launching the dashboard.

## Step 8. Launch SmartCart

From the project root directory, execute:

```bash
python -m streamlit run app.py
```

Streamlit will start the application and display a local URL in the terminal.

Open your browser and visit:

```text
http://localhost:8501
```

**Congratulations! SmartCart is now running locally on your PC.**

---

## 🛠️ Troubleshooting

### 1. Streamlit command not recognised

Try:

```bash
python -m streamlit run app.py
```

### 2. MongoDB connection error

- Verify that MongoDB Community Server is running.
- Check the `MONGODB_URI` in your `.env` file.
- Verify that the MongoDB server is accessible on port `27017`.

### 3. No records found in the dashboard

- Verify that the `smartcart` database exists.
- Check that the `shopping_sessions` collection contains documents.
- Run the dataset initialization/import script if necessary.

### 4. Missing Python modules

Install the missing dependency:

```bash
pip install package_name
```

Alternatively, reinstall all dependencies:

```bash
pip install -r requirements.txt
```

### 5. Missing trained model files

Check that the following files exist:

```text
ml/saved_models/random_forest_tuned.joblib
ml/saved_models/threshold.json
reports/threshold_comparison.csv
```

The model artifacts must be available before using the prediction and model-performance sections.

### 6. Dashboard port already in use

Start Streamlit on another port:

```bash
python -m streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
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

The tuned threshold is stored in:

```text
ml/saved_models/threshold.json
```

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
