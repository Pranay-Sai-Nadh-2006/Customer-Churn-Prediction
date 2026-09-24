# Customer Churn Prediction and RFM-Based Customer Risk Analysis for Online Retail

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-1.5+-150458?style=flat&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat)

## 📌 Executive Summary

This project delivers an end-to-end **Machine Learning & Customer Risk Intelligence System** developed for high-volume online retail businesses. By synthesizing transaction-level data spanning two years (1,067,371 raw records), the system autonomously:
1. Profiles customer buying habits using **RFM (Recency, Frequency, Monetary) Segmentation**.
2. Extracts 20 behavioral features using a **strict temporal cutoff** to prevent data leakage.
3. Trains and benchmarks multiple classification algorithms (**Logistic Regression, Random Forest, Gradient Boosting**).
4. Predicts individual customer churn probabilities and stratifies customers into **High, Medium, and Low Risk tiers**.
5. Produces a prioritized **Customer Danger List (`customer_danger_list.csv`)** paired with actionable, tailored retention strategies.
6. Generates visual risk dashboards and an individual **Customer Report Card lookup system**.

The complete machine learning pipeline is implemented as a production-grade **Python application (`PranaySaiNadh_CustomerChurnPrediction.py`)**.

---

## 🎯 Business Problem & Objectives

In non-contractual e-commerce settings, customers do not explicitly notify the company when they discontinue shopping—they simply lapse. Manually inspecting millions of transactions to spot lapsing customers is impossible at scale.

**Core Objectives:**
* **Automate Churn Detection:** Transition from reactive panic to proactive customer retention.
* **Eliminate Data Leakage:** Use a time-window split (75% history / 25% observation) so future purchase events are never leaked into feature generation.
* **Prioritize High-Value Accounts:** Identify high-spending champions and loyal customers who show declining purchase frequency before they are permanently lost.
* **Provide Operational Guidance:** Output ready-to-use retention playbooks for marketing and account management teams.

---

## 🏗️ Project Architecture & Workflow

```
Raw Transaction Records (~1.06M rows)
  │
  ├── 1. Data Cleaning & Integrity Filtering
  │      • Drop missing CustomerIDs (243,007 rows)
  │      • Filter cancellations (Invoice starting with 'C')
  │      • Filter non-positive Quantity & UnitPrice (805,549 clean records remain)
  │
  ├── 2. Exploratory Data Analysis (EDA)
  │      • Monthly volume and revenue patterns
  │      • Top revenue-generating products and countries
  │
  ├── 3. RFM Analysis & Customer Segmentation
  │      • Recency (days), Frequency (unique orders), Monetary (total spend)
  │      • Quantile scoring (1–4) across R, F, M
  │      • Segmentation: Champions, Loyal Customers, At Risk, Dormant, etc.
  │
  ├── 4. Behavioral Feature Engineering (Historical Window: 75% Timeline)
  │      • Feature Cutoff: 2011-06-07
  │      • 20 behavioral features: order gaps, lifetime, spend trends, recent 90-day activity
  │
  ├── 5. Leakage-Free Churn Target (Observation Window: 25% Timeline)
  │      • Churn = 1 (Zero purchases in observation window)
  │      • Churn = 0 (At least 1 purchase in observation window)
  │      • Base churn rate: 47.7%
  │
  ├── 6. ML Model Training & Benchmarking (Stratified 80/20 Split)
  │      • Logistic Regression | Random Forest | Gradient Boosting
  │      • Preprocessing Pipeline: Median/StandardScaler + Mode/OrdinalEncoder
  │      • Balanced class weights for optimal minority-class recall
  │
  ├── 7. Scoring & Retention Intelligence
  │      • Compute churn probability [0, 1] for all customers
  │      • Risk tier assignment (High >= 70%, Medium 40-70%, Low < 40%)
  │      • Export `customer_danger_list.csv` with prescribed retention actions
  │
  └── 8. Reporting & Dashboards
         • Interactive customer report card generator
         • 6-panel executive customer risk dashboard
```

---

## 📊 Model Evaluation & Benchmark Results

All models were evaluated on an independent, stratified 20% test set (992 customers) preserving the 47.7% churn class balance:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Selected) | **70.56%** | **67.30%** | **74.42%** | **0.7068** | **0.7890** | **0.7472** |
| **Random Forest** | 71.77% | 70.06% | 71.25% | 0.7065 | 0.7928 | 0.7587 |
| **Gradient Boosting** | 71.37% | 69.57% | 71.04% | 0.7029 | 0.8010 | 0.7692 |

### 🏆 Model Selection Rationale
* **Primary Metric — F1-Score (0.7068) & Recall (74.42%):** In customer retention, false negatives (failing to detect a churner) are significantly more costly than false positives (reaching out to a customer who wasn't leaving). Logistic Regression achieved the highest churn recall (**74.42%**), capturing nearly 3 out of every 4 churning customers while maintaining stable calibration across probability thresholds.

---

## 👥 Customer Risk Stratification

From **4,956 unique customers** active in the historical modeling period:

| Risk Tier | Probability Range | Customer Count | Percentage | Retention Playbook |
| :--- | :---: | :---: | :---: | :--- |
| 🔴 **High Risk** | $\ge 70\%$ | **1,483** | **29.9%** | Immediate executive outreach for top accounts; win-back discounts (15–20% off) for preferred categories; automated re-engagement surveys. |
| 🟡 **Medium Risk** | $40\% - 70\%$ | **1,662** | **33.5%** | Proactive product recommendations; early access to sales; free shipping incentives on next purchase. |
| 🟢 **Low Risk** | $< 40\%$ | **1,811** | **36.5%** | Standard promotional newsletters; loyalty point accumulation; referral rewards. |

---

## 📁 Repository Structure

```
Customer_Churn_Prediction/
│
├── PranaySaiNadh_CustomerChurnPrediction.py     # Standalone end-to-end Python pipeline
├── PranaySaiNadh_ProjectReport.docx             # Academic / internship project report document
├── README.md                                    # Comprehensive project documentation
├── requirements.txt                             # Tested, compatible Python dependencies
├── customer_danger_list.csv                     # Scored customers with retention recommendations
│
├── eda_overview.png                             # Monthly volume & revenue trends
├── eda_customer.png                             # Spend & order frequency distributions
├── rfm_segments.png                             # RFM customer segmentation distributions
├── churn_distribution.png                       # Churn balance in observation window
├── confusion_matrices.png                       # Test-set confusion matrices for all models
├── roc_pr_curves.png                            # ROC and Precision-Recall evaluation curves
├── risk_analysis.png                            # 6-panel executive customer risk dashboard
└── online_retail_II.csv                         # Online Retail II dataset (94.8 MB)
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure **Python 3.9+** is installed on your operating system:
```bash
python --version
```

### 2. Clone / Open Directory
```bash
cd c:/Customer_Churn_Prediction
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Project

Run the complete pipeline from the command line:
```bash
python PranaySaiNadh_CustomerChurnPrediction.py
```
This single command automatically:
- Ingests and cleans the transaction records.
- Computes and visualizes EDA and RFM segmentation.
- Performs temporal feature engineering without data leakage.
- Trains, evaluates, and compares Logistic Regression, Random Forest, and Gradient Boosting.
- Scores all customers and generates `customer_danger_list.csv`.
- Generates all visualization charts (`*.png`) and displays an executive performance summary.

---

## 📋 Sample Customer Report Card Demo

When querying a customer (e.g., CustomerID `12346`):

```text
============================================================
  CUSTOMER REPORT CARD
============================================================
  Customer ID       : 12346
  RFM Segment       : Loyal Customers
------------------------------------------------------------
  Recency           : 139 days since last purchase
  Frequency         : 12 unique orders
  Monetary          : £77,556.46 lifetime spend
  RFM Score         : 10 / 12
------------------------------------------------------------
  Churn Probability : 100.0%
  Risk Level        : [HIGH] High Risk
------------------------------------------------------------
  Suggested Action  :
    Urgent Priority: Personal executive outreach,
    exclusive loyalty re-engagement offer, and dedicated
    account manager call.
============================================================
```

---

## 🔮 Future Scope & Enhancements

1. **Real-Time Scoring API:** Containerize the pipeline with FastAPI and Docker for integration with e-commerce checkout webhooks.
2. **Customer Lifetime Value (CLV):** Marry churn probability with a Gamma-Gamma monetary model to estimate monetary value at risk.
3. **Advanced Architectures:** Benchmark gradient boosted trees (LightGBM, CatBoost) with Bayesian hyperparameter tuning.
4. **CRM Integration:** Directly push high-risk customer segments into marketing tools (HubSpot, Salesforce, Klaviyo).

---
