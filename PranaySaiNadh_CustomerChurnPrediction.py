"""
================================================================================
Customer Churn Prediction and RFM-Based Customer Risk Analysis for Online Retail
================================================================================

Student Name  : G. Pranay Sai Nadh
College       : Mohan Babu University
Project Type  : B.Tech Final-Year Machine Learning / Data Science Internship Project
Dataset       : Online Retail II (UCI Machine Learning Repository)
Dataset Link  : https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

Project Workflow:
  1. Dataset Loading & Understanding
  2. Data Cleaning & Integrity Handling
  3. Exploratory Data Analysis (EDA)
  4. RFM (Recency, Frequency, Monetary) Customer Segmentation
  5. Predictive Behavioural Feature Engineering (strictly historical)
  6. Leakage-Free Churn Label Creation (future observation window)
  7. Stratified Train / Test Split & Scikit-Learn Pipelines
  8. Model Training & Evaluation (Logistic Regression, Random Forest, Gradient Boosting)
  9. Best Model Selection & Churn Risk Probability Scoring
 10. Customer Danger List Export (customer_danger_list.csv)
 11. Individual Customer Report Card Generator
 12. Visual Risk Analysis & Final Executive Summary
================================================================================
"""

import sys
import warnings
from datetime import timedelta
from pathlib import Path
import textwrap

# Ensure UTF-8 output encoding across all Windows shells
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/script execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# Global Configuration
# ──────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def find_dataset(explicit_path=None) -> Path:
    """Locate the Online Retail II dataset file (.csv or .xlsx)."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified dataset path not found: {explicit_path}")

    search_dir = Path(".")
    candidates = list(search_dir.glob("*online*retail*.*")) + list(search_dir.glob("*.csv"))
    valid_exts = {".csv", ".xlsx", ".xls"}

    for c in candidates:
        if c.suffix.lower() in valid_exts and "checkpoint" not in str(c):
            return c

    raise FileNotFoundError("Could not auto-detect dataset. Please ensure online_retail_II.csv exists.")


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the dataset handling CSV or Excel formats and clean column names."""
    print("=" * 70)
    print(f"LOADING DATASET: {dataset_path.name}")
    print("=" * 70)

    if dataset_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path, encoding="utf-8-sig", low_memory=False)
    else:
        xls = pd.ExcelFile(dataset_path)
        sheets = [pd.read_excel(xls, sheet) for sheet in xls.sheet_names]
        df = pd.concat(sheets, ignore_index=True)

    col_mapping = {
        "Customer ID": "CustomerID",
        "Price": "UnitPrice",
        "Invoice": "Invoice",
        "StockCode": "StockCode",
        "Description": "Description",
        "Quantity": "Quantity",
        "InvoiceDate": "InvoiceDate",
        "Country": "Country",
    }
    df.rename(columns=col_mapping, inplace=True)
    print(f"  Raw records loaded : {len(df):,}")
    print(f"  Columns            : {list(df.columns)}")
    return df


def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Perform data cleaning: remove missing CustomerID, cancellations, negative quantities & prices."""
    print("\n" + "=" * 70)
    print("DATA CLEANING")
    print("=" * 70)
    rows_before = len(df_raw)

    df = df_raw.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df[df["InvoiceDate"].notna()]

    # 1. Drop records with missing CustomerID
    missing_cust = df["CustomerID"].isna().sum()
    df = df[df["CustomerID"].notna()].copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    # 2. Filter out cancellations (Invoice starting with 'C')
    cancelled = df["Invoice"].astype(str).str.strip().str.upper().str.startswith("C")
    df = df[~cancelled].copy()

    # 3. Keep only positive quantity and unit price
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

    # 4. Total Amount per transaction line
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    rows_after = len(df)
    print(f"  Records before cleaning : {rows_before:,}")
    print(f"  Missing Customer IDs    : {missing_cust:,} removed")
    print(f"  Records after cleaning  : {rows_after:,} (removed {rows_before - rows_after:,} total)")
    print(f"  Unique Customers        : {df['CustomerID'].nunique():,}")
    print(f"  Date Range              : {df['InvoiceDate'].min().date()} -> {df['InvoiceDate'].max().date()}")
    return df


def generate_eda_plots(df: pd.DataFrame):
    """Generate and save Exploratory Data Analysis figures."""
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Monthly Transactions and Revenue
    df_temp = df.copy()
    df_temp["YearMonth"] = df_temp["InvoiceDate"].dt.to_period("M")
    monthly_stats = df_temp.groupby("YearMonth").agg(
        Transactions=("Invoice", "count"),
        Revenue=("TotalAmount", "sum")
    ).reset_index()
    monthly_stats["YearMonth_dt"] = monthly_stats["YearMonth"].dt.to_timestamp()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Monthly Transactions
    ax = axes[0, 0]
    ax.bar(monthly_stats["YearMonth_dt"], monthly_stats["Transactions"], width=20, color=sns.color_palette("muted")[0])
    ax.set_title("Monthly Transaction Volume", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Transactions")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    # Plot 2: Monthly Revenue
    ax = axes[0, 1]
    ax.plot(monthly_stats["YearMonth_dt"], monthly_stats["Revenue"], marker="o", color=sns.color_palette("muted")[1], lw=2)
    ax.fill_between(monthly_stats["YearMonth_dt"], monthly_stats["Revenue"], alpha=0.25, color=sns.color_palette("muted")[1])
    ax.set_title("Monthly Revenue (GBP)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (£)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    # Plot 3: Top 10 Countries by Revenue
    top_countries = df.groupby("Country")["TotalAmount"].sum().sort_values(ascending=False).head(10).reset_index()
    ax = axes[1, 0]
    sns.barplot(data=top_countries, y="Country", x="TotalAmount", palette="muted", ax=ax)
    ax.set_title("Top 10 Countries by Revenue", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Revenue (£)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    # Plot 4: Top 10 Products by Revenue
    top_products = df.groupby("Description")["TotalAmount"].sum().sort_values(ascending=False).head(10).reset_index()
    ax = axes[1, 1]
    sns.barplot(data=top_products, y="Description", x="TotalAmount", palette="muted", ax=ax)
    ax.set_title("Top 10 Products by Revenue", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Revenue (£)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    plt.suptitle("Exploratory Data Analysis – Online Retail II", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("eda_overview.png", bbox_inches="tight", dpi=100)
    plt.close()
    print("  Saved: eda_overview.png")

    # 2. Customer-level EDA
    customer_spend = df.groupby("CustomerID")["TotalAmount"].sum().reset_index(name="TotalSpend")
    customer_orders = df.groupby("CustomerID")["Invoice"].nunique().reset_index(name="NumOrders")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(np.log1p(customer_spend["TotalSpend"]), bins=50, color=sns.color_palette("muted")[2], edgecolor="white")
    ax.set_title("Customer Total Spend (log scale)", fontweight="bold")
    ax.set_xlabel("log(1 + Total Spend £)")
    ax.set_ylabel("Customer Count")

    ax = axes[1]
    ax.hist(customer_orders["NumOrders"].clip(upper=50), bins=40, color=sns.color_palette("muted")[3], edgecolor="white")
    ax.set_title("Order Frequency Distribution", fontweight="bold")
    ax.set_xlabel("Number of Orders")
    ax.set_ylabel("Customer Count")

    top15_customers = customer_spend.sort_values("TotalSpend", ascending=False).head(15)
    ax = axes[2]
    ax.barh(top15_customers["CustomerID"].astype(str)[::-1], top15_customers["TotalSpend"].iloc[::-1].values,
            color=sns.color_palette("muted")[4])
    ax.set_title("Top 15 Customers by Lifetime Spend", fontweight="bold")
    ax.set_xlabel("Total Spend (£)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    plt.tight_layout()
    plt.savefig("eda_customer.png", bbox_inches="tight", dpi=100)
    plt.close()
    print("  Saved: eda_customer.png")


def quantile_score(series: pd.Series, n=4, ascending=True) -> pd.Series:
    """Assign 1-n quantile score safely handling duplicate bins via ranking."""
    labels = list(range(1, n + 1)) if ascending else list(range(n, 0, -1))
    ranked = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranked, q=n, labels=labels).astype(int)


def assign_rfm_segment(row: pd.Series) -> str:
    """Map RFM scores into business customer segments."""
    r, f, m, score = row["R_Score"], row["F_Score"], row["M_Score"], row["RFM_Score"]
    if r >= 3 and f >= 3 and m >= 3:
        return "Champions"
    elif f >= 3 and m >= 3:
        return "Loyal Customers"
    elif r >= 3 and f >= 2:
        return "Potential Loyalists"
    elif r >= 3:
        return "Recent Customers"
    elif r <= 2 and (f >= 3 or m >= 3):
        return "At Risk"
    elif score <= 5:
        return "Dormant / Lost"
    else:
        return "Needs Attention"


def compute_rfm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary scores and customer segments."""
    print("\n" + "=" * 70)
    print("RFM ANALYSIS & CUSTOMER SEGMENTATION")
    print("=" * 70)

    ref_date = df["InvoiceDate"].max() + timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalAmount", "sum")
    ).reset_index()

    rfm["Recency"] = (ref_date - rfm["LastPurchaseDate"]).dt.days
    rfm.drop(columns=["LastPurchaseDate"], inplace=True)

    rfm["R_Score"] = quantile_score(rfm["Recency"], ascending=False)
    rfm["F_Score"] = quantile_score(rfm["Frequency"], ascending=True)
    rfm["M_Score"] = quantile_score(rfm["Monetary"], ascending=True)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    rfm["RFM_Segment"] = rfm.apply(assign_rfm_segment, axis=1)

    print(f"  RFM Table Shape : {rfm.shape[0]:,} customers × {rfm.shape[1]} columns")
    print("  Customer Counts by Segment:")
    for seg, count in rfm["RFM_Segment"].value_counts().items():
        print(f"    • {seg:<20}: {count:,}")

    # Plot RFM Segments
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    seg_counts = rfm["RFM_Segment"].value_counts()
    seg_counts.plot(kind="bar", ax=ax, color=sns.color_palette("muted", len(seg_counts)), edgecolor="white")
    ax.set_title("Customer Count per RFM Segment", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 1]
    rfm["RFM_Score"].value_counts().sort_index().plot(kind="bar", ax=ax, color=sns.color_palette("muted")[1], edgecolor="white")
    ax.set_title("Distribution of RFM Scores (3–12)", fontweight="bold")
    ax.set_xlabel("RFM Score")

    ax = axes[1, 0]
    for i, seg in enumerate(rfm["RFM_Segment"].unique()):
        sub = rfm[rfm["RFM_Segment"] == seg]
        ax.scatter(sub["Recency"], sub["Frequency"].clip(upper=200), label=seg, alpha=0.5, s=15)
    ax.set_title("Recency vs Frequency by Segment", fontweight="bold")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Frequency (orders)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    avg_m = rfm.groupby("RFM_Segment")["Monetary"].median().sort_values(ascending=False)
    avg_m.plot(kind="barh", ax=ax, color=sns.color_palette("muted", len(avg_m)), edgecolor="white")
    ax.set_title("Median Monetary Value by Segment (£)", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    plt.suptitle("RFM Customer Segmentation Analysis", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("rfm_segments.png", bbox_inches="tight", dpi=100)
    plt.close()
    print("  Saved: rfm_segments.png")
    return rfm


def build_predictive_features(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Build predictive customer behavioral features strictly from transactions prior to cutoff_date."""
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING (TEMPORAL CUTOFF - NO DATA LEAKAGE)")
    print("=" * 70)
    print(f"  Feature Cutoff Date : {cutoff_date.date()} (Historical Window)")

    hist = df[df["InvoiceDate"] < cutoff_date].copy()
    recent_start = cutoff_date - timedelta(days=90)

    # 1. Base aggregations
    agg = hist.groupby("CustomerID").agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        FirstPurchaseDate=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalAmount", "sum"),
        TotalQuantity=("Quantity", "sum"),
        NumUniqueProducts=("StockCode", "nunique"),
        Country=("Country", lambda x: x.mode()[0] if len(x) > 0 else "Unknown")
    ).reset_index()

    agg["Recency"] = (cutoff_date - agg["LastPurchaseDate"]).dt.days
    agg["CustomerLifetimeDays"] = (agg["LastPurchaseDate"] - agg["FirstPurchaseDate"]).dt.days
    agg["AvgOrderValue"] = agg["Monetary"] / agg["Frequency"]

    # 2. Active calendar months
    active_m = hist.groupby("CustomerID")["InvoiceDate"].apply(lambda x: x.dt.to_period("M").nunique()).reset_index()
    active_m.columns = ["CustomerID", "ActiveMonths"]
    agg = agg.merge(active_m, on="CustomerID", how="left")

    # 3. Inter-order gap statistics
    def calc_gaps(s):
        dates = s.drop_duplicates().sort_values()
        if len(dates) < 2:
            return pd.Series({"AvgDaysBetweenOrders": np.nan, "StdDaysBetweenOrders": np.nan})
        diffs = dates.diff().dt.days.dropna()
        return pd.Series({"AvgDaysBetweenOrders": diffs.mean(), "StdDaysBetweenOrders": diffs.std()})

    gaps = hist.groupby("CustomerID")["InvoiceDate"].apply(calc_gaps).unstack().reset_index()
    agg = agg.merge(gaps, on="CustomerID", how="left")

    # 4. Recent 90-day activity
    recent_tx = hist[hist["InvoiceDate"] >= recent_start].groupby("CustomerID").agg(
        RecentOrderCount_90d=("Invoice", "nunique"),
        RecentSpend_90d=("TotalAmount", "sum")
    ).reset_index()
    agg = agg.merge(recent_tx, on="CustomerID", how="left")
    agg["RecentOrderCount_90d"] = agg["RecentOrderCount_90d"].fillna(0)
    agg["RecentSpend_90d"] = agg["RecentSpend_90d"].fillna(0)

    # 5. Spend Trend (second half spend / first half spend)
    def calc_trend(grp):
        grp = grp.sort_values("InvoiceDate")
        mid = len(grp) // 2
        if mid == 0:
            return np.nan
        h1 = grp.iloc[:mid]["TotalAmount"].sum()
        h2 = grp.iloc[mid:]["TotalAmount"].sum()
        return (h2 / h1) if h1 > 0 else np.nan

    trend = hist.groupby("CustomerID").apply(calc_trend).reset_index(name="SpendTrend")
    agg = agg.merge(trend, on="CustomerID", how="left")

    # 6. Recalculate historical RFM scores
    agg["R_Score"] = quantile_score(agg["Recency"], ascending=False)
    agg["F_Score"] = quantile_score(agg["Frequency"], ascending=True)
    agg["M_Score"] = quantile_score(agg["Monetary"], ascending=True)
    agg["RFM_Score"] = agg["R_Score"] + agg["F_Score"] + agg["M_Score"]
    agg["RFM_Segment"] = agg.apply(assign_rfm_segment, axis=1)

    agg.drop(columns=["LastPurchaseDate", "FirstPurchaseDate"], inplace=True)
    print(f"  Feature Table Built : {agg.shape[0]:,} customers × {agg.shape[1]} features")
    return agg


def create_churn_labels(df: pd.DataFrame, features_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Label customer as Churn=1 if zero purchases made in the future window after cutoff."""
    print("\n" + "=" * 70)
    print("CHURN TARGET CREATION")
    print("=" * 70)

    future_purchases = df[df["InvoiceDate"] >= cutoff_date]
    active_in_future = set(future_purchases["CustomerID"].unique())

    df_labeled = features_df.copy()
    df_labeled["Churn"] = df_labeled["CustomerID"].apply(lambda cid: 0 if cid in active_in_future else 1)

    churn_counts = df_labeled["Churn"].value_counts()
    churn_rate = df_labeled["Churn"].mean()

    print(f"  Non-Churned (0) : {churn_counts.get(0, 0):,}")
    print(f"  Churned     (1) : {churn_counts.get(1, 0):,}")
    print(f"  Base Churn Rate : {churn_rate:.1%}")

    # Plot Churn Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(["Not Churned (0)", "Churned (1)"], [churn_counts.get(0, 0), churn_counts.get(1, 0)],
                color=[sns.color_palette("muted")[0], sns.color_palette("muted")[3]])
    axes[0].set_title("Churn Distribution", fontweight="bold")
    axes[0].set_ylabel("Customers")

    axes[1].pie([churn_counts.get(0, 0), churn_counts.get(1, 0)], labels=["Not Churned", "Churned"],
                colors=[sns.color_palette("muted")[0], sns.color_palette("muted")[3]],
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Churn Proportion", fontweight="bold")
    plt.tight_layout()
    plt.savefig("churn_distribution.png", bbox_inches="tight", dpi=100)
    plt.close()
    print("  Saved: churn_distribution.png")
    return df_labeled


def train_and_evaluate_models(model_df: pd.DataFrame):
    """Train Logistic Regression, Random Forest, and Gradient Boosting pipelines and evaluate."""
    print("\n" + "=" * 70)
    print("MACHINE LEARNING MODEL TRAINING & EVALUATION")
    print("=" * 70)

    numeric_features = [
        "Recency", "Frequency", "Monetary", "AvgOrderValue",
        "TotalQuantity", "NumUniqueProducts", "ActiveMonths",
        "AvgDaysBetweenOrders", "StdDaysBetweenOrders",
        "CustomerLifetimeDays", "RecentOrderCount_90d", "RecentSpend_90d",
        "SpendTrend", "R_Score", "F_Score", "M_Score", "RFM_Score"
    ]
    categorical_features = ["Country"]
    target = "Churn"

    # Cardinality reduction for Country (keep top 15)
    top_countries = model_df["Country"].value_counts().head(15).index.tolist()
    data = model_df.copy()
    data["Country"] = data["Country"].apply(lambda c: c if c in top_countries else "Other")

    X = data[numeric_features + categorical_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training Set : {len(X_train):,} samples ({y_train.mean():.1%} churn)")
    print(f"  Test Set     : {len(X_test):,} samples ({y_test.mean():.1%} churn)")

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical_features)
    ])

    models = {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", preprocessor),
            ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=RANDOM_STATE))
        ])
    }

    results = []
    trained_pipelines = {}

    for name, pipe in models.items():
        print(f"  Training {name}...", end=" ", flush=True)
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc,
            "PR-AUC": pr_auc
        })
        print("done.")

    results_df = pd.DataFrame(results).set_index("Model")
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON ON TEST SET")
    print("=" * 70)
    print(results_df.round(4).to_string())

    best_model_name = results_df["F1-Score"].idxmax()
    print(f"\n  BEST MODEL SELECTED (by F1-Score): {best_model_name}")

    # Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, pipe) in enumerate(trained_pipelines.items()):
        cm = confusion_matrix(y_test, pipe.predict(X_test))
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[i],
                    xticklabels=["Active (0)", "Churn (1)"], yticklabels=["Active (0)", "Churn (1)"])
        axes[i].set_title(f"Confusion Matrix: {name}", fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", bbox_inches="tight", dpi=100)
    plt.close()
    print("  Saved: confusion_matrices.png")

    # ROC and PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, pipe in trained_pipelines.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_prob):.3f})", lw=2)

        p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
        axes[1].plot(r_curve, p_curve, label=f"{name} (AP = {average_precision_score(y_test, y_prob):.3f})", lw=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("ROC Curves", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].set_title("Precision-Recall Curves", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("roc_pr_curves.png", bbox_inches="tight", dpi=100)
    plt.close()
    print("  Saved: roc_pr_curves.png")

    return trained_pipelines[best_model_name], best_model_name, results_df, X, y


def suggest_retention_action(row: pd.Series) -> str:
    """Provide tailored retention action based on risk level and customer segment."""
    risk = row["Risk_Level"]
    segment = row["RFM_Segment"]

    if risk == "High Risk":
        if segment in ("Champions", "Loyal Customers"):
            return "Urgent Priority: Personal executive outreach, exclusive loyalty re-engagement offer, and dedicated account manager call."
        elif segment == "At Risk":
            return "Re-activation campaign: Win-back discount code (15-20% off) for preferred product categories and email reminder."
        else:
            return "Automated win-back sequence: 'We miss you' promotional voucher and customer survey on shopping experience."
    elif risk == "Medium Risk":
        if segment in ("Champions", "Loyal Customers", "Potential Loyalists"):
            return "Proactive engagement: Personalized product recommendations and early access to upcoming sales."
        else:
            return "Targeted newsletter highlighting trending products and low-threshold free shipping incentive."
    else:
        return "Maintenance & Loyalty: Standard promotional updates, reward point accumulation, and referral incentives."


def generate_customer_danger_list(best_pipeline, model_df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Score all customers, rank by churn risk probability, and export customer_danger_list.csv."""
    print("\n" + "=" * 70)
    print("CUSTOMER CHURN PROBABILITY SCORING & DANGER LIST")
    print("=" * 70)

    probs = best_pipeline.predict_proba(X)[:, 1]
    danger_df = model_df[["CustomerID", "Recency", "Frequency", "Monetary", "RFM_Score", "RFM_Segment"]].copy()
    danger_df["Churn_Probability"] = probs

    def map_risk(p):
        if p >= HIGH_RISK_THRESHOLD:
            return "High Risk"
        elif p >= MEDIUM_RISK_THRESHOLD:
            return "Medium Risk"
        return "Low Risk"

    danger_df["Risk_Level"] = danger_df["Churn_Probability"].apply(map_risk)
    danger_df["Suggested_Action"] = danger_df.apply(suggest_retention_action, axis=1)

    danger_df.sort_values("Churn_Probability", ascending=False, inplace=True)

    csv_path = "customer_danger_list.csv"
    danger_df.to_csv(csv_path, index=False)
    print(f"  Customer Danger List exported : {csv_path}")
    print(f"  Total Customers Scored        : {len(danger_df):,}")

    risk_counts = danger_df["Risk_Level"].value_counts()
    print("  Customer Risk Breakdown:")
    for risk, cnt in risk_counts.items():
        print(f"    • {risk:<12}: {cnt:,} ({cnt/len(danger_df):.1%})")

    return danger_df


def display_customer_report_card(danger_df: pd.DataFrame, customer_id: str):
    """Print an individual customer report card showing metrics and retention actions."""
    cid = str(customer_id).strip()
    match = danger_df[danger_df["CustomerID"] == cid]

    if match.empty:
        print(f"[!] CustomerID '{cid}' not found in the customer database.")
        return

    row = match.iloc[0]
    icons = {"High Risk": "[HIGH]", "Medium Risk": "[MED]", "Low Risk": "[LOW]"}
    risk_icon = icons.get(row["Risk_Level"], "[INFO]")

    print("\n" + "=" * 60)
    print("  CUSTOMER REPORT CARD")
    print("=" * 60)
    print(f"  Customer ID       : {row['CustomerID']}")
    print(f"  RFM Segment       : {row['RFM_Segment']}")
    print("-" * 60)
    print(f"  Recency           : {row['Recency']:.0f} days since last purchase")
    print(f"  Frequency         : {row['Frequency']:.0f} unique orders")
    print(f"  Monetary          : £{row['Monetary']:,.2f} lifetime spend")
    print(f"  RFM Score         : {row['RFM_Score']:.0f} / 12")
    print("-" * 60)
    print(f"  Churn Probability : {row['Churn_Probability']:.1%}")
    print(f"  Risk Level        : {risk_icon} {row['Risk_Level']}")
    print("-" * 60)
    print("  Suggested Action  :")
    for line in textwrap.wrap(row["Suggested_Action"], width=54):
        print(f"    {line}")
    print("=" * 60)


def generate_risk_dashboard(danger_df: pd.DataFrame):
    """Generate 6-panel visual customer risk analysis dashboard."""
    print("\n" + "=" * 70)
    print("GENERATING VISUAL RISK ANALYSIS DASHBOARD")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Churn probability histogram
    ax = axes[0, 0]
    ax.hist(danger_df["Churn_Probability"], bins=30, color=sns.color_palette("muted")[0], edgecolor="white")
    ax.axvline(HIGH_RISK_THRESHOLD, color="red", linestyle="--", label=f"High Risk (≥{HIGH_RISK_THRESHOLD:.0%})")
    ax.axvline(MEDIUM_RISK_THRESHOLD, color="orange", linestyle="--", label=f"Med Risk (≥{MEDIUM_RISK_THRESHOLD:.0%})")
    ax.set_title("Churn Probability Distribution", fontweight="bold")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Customers")
    ax.legend(fontsize=8)

    # 2. Customers by Risk Level
    ax = axes[0, 1]
    risk_counts = danger_df["Risk_Level"].value_counts()[["High Risk", "Medium Risk", "Low Risk"]]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    risk_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Customer Count by Risk Level", fontweight="bold")
    ax.set_ylabel("Customers")
    ax.tick_params(axis="x", rotation=0)

    # 3. Churn probability by RFM segment
    ax = axes[0, 2]
    sns.boxplot(data=danger_df, x="RFM_Segment", y="Churn_Probability", ax=ax, palette="muted")
    ax.set_title("Churn Probability by RFM Segment", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    # 4. Monetary vs Recency scatter colored by churn probability
    ax = axes[1, 0]
    sc = ax.scatter(danger_df["Recency"], danger_df["Monetary"].clip(upper=10000),
                    c=danger_df["Churn_Probability"], cmap="coolwarm", alpha=0.6, s=15)
    plt.colorbar(sc, ax=ax, label="Churn Probability")
    ax.set_title("Monetary vs Recency (colored by risk)", fontweight="bold")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary (£)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    # 5. Risk Level composition within segments
    ax = axes[1, 1]
    seg_risk = pd.crosstab(danger_df["RFM_Segment"], danger_df["Risk_Level"], normalize="index")[["High Risk", "Medium Risk", "Low Risk"]]
    seg_risk.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white")
    ax.set_title("Risk Level Composition per Segment", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)

    # 6. High Risk Customer Revenue at Risk
    ax = axes[1, 2]
    rev_risk = danger_df.groupby("Risk_Level")["Monetary"].sum()[["High Risk", "Medium Risk", "Low Risk"]]
    rev_risk.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Total Revenue by Risk Category (£)", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.tick_params(axis="x", rotation=0)

    plt.suptitle("Customer Churn Risk & Retention Dashboard", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("risk_analysis.png", bbox_inches="tight", dpi=100)
    plt.close()
    print("  Saved: risk_analysis.png")


def main():
    """Main execution orchestrating all project stages end-to-end."""
    print("*" * 75)
    print("CUSTOMER CHURN PREDICTION & RETENTION RISK SCORING PIPELINE")
    print("*" * 75)

    # 1. Dataset Discovery & Loading
    dataset_file = find_dataset()
    raw_df = load_dataset(dataset_file)

    # 2. Data Cleaning
    clean_df = clean_dataset(raw_df)

    # 3. Exploratory Data Analysis
    generate_eda_plots(clean_df)

    # 4. RFM Customer Segmentation
    compute_rfm_analysis(clean_df)

    # 5. Temporal Splitting Setup (75% history / 25% observation)
    min_date = clean_df["InvoiceDate"].min()
    max_date = clean_df["InvoiceDate"].max()
    total_days = (max_date - min_date).days
    feature_cutoff = min_date + timedelta(days=int(total_days * 0.75))

    # 6. Predictive Feature Engineering & Churn Target
    features_df = build_predictive_features(clean_df, feature_cutoff)
    labeled_df = create_churn_labels(clean_df, features_df, feature_cutoff)

    # 7. Model Training & Evaluation
    best_pipe, best_name, results_table, X, y = train_and_evaluate_models(labeled_df)

    # 8. Churn Probability Scoring & Danger List
    danger_df = generate_customer_danger_list(best_pipe, labeled_df, X)

    # 9. Individual Customer Report Card Demo
    top_risk_customer = danger_df.iloc[0]["CustomerID"]
    print("\nDemonstration: Individual Customer Report Card for Highest Risk Customer")
    display_customer_report_card(danger_df, top_risk_customer)

    # 10. Risk Analysis Dashboard
    generate_risk_dashboard(danger_df)

    # 11. Final Executive Summary
    print("\n" + "=" * 75)
    print("  PROJECT PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    print(f"  • Best Performing Model      : {best_name}")
    print(f"  • Best Model F1-Score        : {results_table.loc[best_name, 'F1-Score']:.4f}")
    print(f"  • Best Model ROC-AUC         : {results_table.loc[best_name, 'ROC-AUC']:.4f}")
    print(f"  • Total Customers Analyzed   : {len(danger_df):,}")
    print(f"  • High Risk (Action Needed)  : {(danger_df['Risk_Level'] == 'High Risk').sum():,} customers")
    print(f"  • Medium Risk Customers      : {(danger_df['Risk_Level'] == 'Medium Risk').sum():,} customers")
    print(f"  • Low Risk (Loyal/Engaged)   : {(danger_df['Risk_Level'] == 'Low Risk').sum():,} customers")
    print("  • Outputs Generated:")
    print("      1. customer_danger_list.csv (Prioritized Retention Actions)")
    print("      2. eda_overview.png")
    print("      3. eda_customer.png")
    print("      4. rfm_segments.png")
    print("      5. churn_distribution.png")
    print("      6. confusion_matrices.png")
    print("      7. roc_pr_curves.png")
    print("      8. risk_analysis.png")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
