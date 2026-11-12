# SmartCart – E-Commerce Purchase Intention Analytics

An end-to-end NoSQL and Machine Learning project that analyses e-commerce shopping behaviour, identifies purchase patterns, and predicts customer purchase intention using MongoDB, Python, and a tuned Random Forest model.

## 1. Project Overview

SmartCart is an e-commerce analytics platform built using MongoDB and Streamlit. It analyses online shopping sessions to understand customer browsing behaviour and identify patterns that influence purchasing decisions.

The project integrates:

- **MongoDB** – NoSQL database for storing shopping session data.
- **PyMongo** – Python driver for MongoDB connectivity.
- **Python** – Data processing and machine learning.
- **Pandas & NumPy** – Data manipulation and analysis.
- **Scikit-learn** – Machine learning model development and evaluation.
- **Streamlit** – Interactive analytics dashboard.
- **Plotly** – Interactive data visualizations.
- **Joblib** – Saving and loading trained machine learning models.

The application provides three main modules:

1. **Business Analytics** – Analyses customer sessions, purchase conversion, visitor types, and browsing behaviour.
2. **Purchase Prediction** – Uses a tuned Random Forest classifier to predict purchase intention.
3. **Model Performance** – Presents machine learning evaluation metrics.

## 2. Dataset Information

**Dataset:** Online Shoppers Purchasing Intention Dataset

| Attribute | Value |
|---|---|
| Total Records | 12,330 |
| Total Features | 18 |
| Target Variable | Revenue |
| Purchase Sessions | 1,908 |
| Non-Purchase Sessions | 10,422 |
| Purchase Conversion Rate | 15.47% |

The dataset contains information about visitor types, browsing duration, page visits, bounce rates, exit rates, page values, operating systems, browser information, and other shopping session attributes.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Database | MongoDB |
| Database Connectivity | PyMongo |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Configuration | python-dotenv |

## 4. Project Architecture

The overall workflow follows this structure:

```text
Online Shoppers Dataset
          |
          v
    Data Processing
          |
          v
   MongoDB Database
          |
          v
     PyMongo Driver
          |
          v
   MongoDB Aggregation
          |
          v
    Python Analytics
          |
          +----------------------+
          |                      |
          v                      v
   Business Analytics     Saved Random Forest
                                 |
                                 v
                         Purchase Prediction
          |                      |
          +----------------------+
                     |
                     v
             Streamlit Dashboard
                     |
                     v
             Interactive Results
```

## 5. MongoDB Implementation

MongoDB is used as the primary database to store and retrieve shopping session records.

### Database Structure

```text
MongoDB Server
    |
    └── smartcart
           |
           └── shopping_sessions
                    |
                    └── 12,330 Documents
```

**Database Name:** `smartcart`

**Collection Name:** `shopping_sessions`

### MongoDB Operations Implemented

The project uses the following MongoDB operations:

| Operation | Purpose |
|---|---|
| `insertMany()` | Import multiple shopping session documents |
| `find()` | Retrieve documents |
| `countDocuments()` | Count matching documents |
| `aggregate()` | Perform analytical operations |
| `$group` | Group documents by specified fields |
| `$sum` | Calculate totals |
| `$avg` | Calculate averages |
| `$cond` | Perform conditional calculations |
| `$sort` | Sort aggregation results |
| `createIndex()` | Improve query performance |
| Collection Validator | Enforce document structure and data validation |

### Aggregation Analysis

Three major aggregation analyses are implemented in the dashboard:

**1. Visitor Type Analysis**

Groups shopping sessions by visitor type and calculates:
- Total sessions.
- Number of purchases.
- Purchase conversion rate.

**2. Weekend vs Weekday Analysis**

Groups sessions based on the `Weekend` field and compares purchase behaviour.

**3. Browsing Behaviour Analysis**

Groups sessions by purchase outcome (`Revenue`) and calculates:
- Number of sessions.
- Average product-related page visits.
- Average product-related browsing duration.
- Average page values.

## 6. Machine Learning Implementation

The project uses a **Random Forest Classifier** to predict whether a visitor is likely to make a purchase.

### Model Development

The machine learning workflow includes:

1. Data preprocessing.
2. Feature preparation.
3. Training and validation.
4. Random Forest classification.
5. Hyperparameter tuning.
6. Decision threshold optimization.
7. Model evaluation.
8. Saving the trained model using Joblib.

### Model Results

Recorded project evaluation results:

| Metric | Result |
|---|---:|
| Random Forest Accuracy | 85.08% |
| Random Forest ROC-AUC | 0.7946 |
| Selected Decision Threshold | Approximately 0.154 |
| Validation F1-Score at Tuned Threshold | 0.3969 |
| Test Recall at Tuned Threshold | 0.8037 |

The decision threshold was adjusted to improve the model's ability to identify potential purchasing sessions.

The trained model and threshold metadata are stored separately:

```text
ml/
   saved_models/
       random_forest_tuned.joblib
       threshold.json
```

The dashboard loads these saved files instead of retraining the model every time the application starts.

## 7. Dashboard Features

### Business Analytics

- Total shopping sessions.
- Purchase and non-purchase sessions.
- Overall conversion rate.
- Session distribution by visitor type.
- Conversion rate by visitor type.
- Weekday vs weekend purchase comparison.
- Browsing behaviour by purchase outcome.

### Purchase Prediction

Uses the saved Random Forest model to estimate purchase intention based on shopping session features.

### Model Performance

Displays model evaluation information to help understand the classifier's predictive performance.

## 8. Installation and Setup

Follow these instructions to run SmartCart on another computer.

### Prerequisites

Install the following software:

- Python (preferably the same version used during development).
- MongoDB Community Server.
- Git.
- MongoDB Compass (optional, for graphical database management).

### Step 1: Clone the Repository

Open PowerShell or a terminal:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project directory:

```bash
cd "NoSql Project"
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The dependency list should include the compatible versions of:

```text
pymongo
streamlit
pandas
numpy
plotly
scikit-learn
joblib
python-dotenv
```

**Important:** Use a tested `requirements.txt` with compatible package versions, particularly for Scikit-learn, because the saved model was serialized using a specific software environment.

### Step 4: Configure MongoDB

Make sure MongoDB Community Server is installed and running.

The default local MongoDB connection URI is:

```text
mongodb://localhost:27017/
```

Create a `.env` file in the root project directory:

```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=smartcart
MONGODB_COLLECTION=shopping_sessions
```

The application reads these environment variables using `python-dotenv`.

### Step 5: Import the Database

The application requires the `smartcart` database and its `shopping_sessions` collection to contain the dataset.

**Important:** Cloning the GitHub repository does not automatically transfer your local MongoDB database.

You must import the dataset on the new computer.

You can either:

- Import `online_shoppers_intention.csv` using the project's database import script.
- Restore an exported MongoDB database backup using MongoDB Database Tools.

For a standard MongoDB dump containing the `smartcart` folder:

```bash
mongorestore --db smartcart ./dump/smartcart
```

Make sure the collection name is `shopping_sessions`.

Also ensure that Boolean fields such as `Revenue` and `Weekend` retain their Boolean data types and numerical features remain numeric.

### Step 6: Verify the Database and Model

Run:

```bash
python check_dashboard.py
```

This diagnostic script checks:

- MongoDB connectivity.
- Database collection access.
- Number of documents.
- Saved Random Forest model loading.
- Selected decision threshold.
- Number of model features.

Expected output includes:

```text
Checking MongoDB connection...
MongoDB connection: OK
Collection documents: 12330

Loading saved model...
Saved model: OK
Selected threshold: 0.154...

All checks completed.
```

The document count may differ if you imported a different dataset or subset.

### Step 7: Run the Dashboard

Start Streamlit:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal:

```text
http://localhost:8501
```

SmartCart should now be running.

## 9. Project Structure

The main application files are organized as follows:

```text
NoSql Project/
│
├── app.py
├── check_dashboard.py
├── .env
├── requirements.txt
│
├── ml/
│   └── saved_models/
│       ├── random_forest_tuned.joblib
│       └── threshold.json
│
├── reports/
│   └── threshold_comparison.csv
│
└── README.md
```

Additional folders may contain the dataset, preprocessing scripts, MongoDB setup scripts, model training code, and project documentation.

## 10. Troubleshooting

| Issue | Solution |
|---|---|
| MongoDB connection refused | Ensure MongoDB Server is running on port 27017. |
| ModuleNotFoundError | Activate the virtual environment and install dependencies. |
| Collection not found or empty | Import the dataset into `smartcart.shopping_sessions`. |
| Model file not found | Verify the files inside `ml/saved_models/`. |
| Streamlit command not recognized | Use `python -m streamlit run app.py`. |
| Port 8501 already in use | Run Streamlit using another port, such as `--server.port 8502`. |
| Model compatibility error | Install a compatible Scikit-learn version. |
| Incorrect analytics results | Verify MongoDB field names and data types. |

## 11. Important GitHub Notes

Before uploading the project, ensure that the following are not committed:

```text
.venv/
__pycache__/
.env
```

A recommended `.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
.streamlit/secrets.toml
```

Do not upload database credentials or private information.

The MongoDB database itself is stored separately from the GitHub repository. To reproduce the project on another computer, the database must be imported or restored independently.

Also ensure that the dataset and trained model files can legally be redistributed before making them public.

## 12. Future Improvements

- Deploy the Streamlit dashboard online.
- Add real-time customer behaviour monitoring.
- Improve prediction performance through additional feature engineering.
- Introduce more advanced MongoDB aggregation analytics.
- Implement automated database backup and recovery.
- Improve dashboard interactivity and reporting.

## Author

**Pranav Sharma**

**Project:** SmartCart – E-Commerce Purchase Intention Analytics

**Domain:** NoSQL Databases | Data Analytics | Machine Learning

---

*SmartCart demonstrates the integration of NoSQL database management, analytical querying, machine learning, and interactive dashboard development in an end-to-end e-commerce analytics application.*
