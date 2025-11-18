# ============================================
# Customer Revenue & Churn Intelligence Pipeline
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tomli
from pymongo import MongoClient

# --------------------------------------------
# 1. LOAD & INSPECT DATA
# --------------------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)

    return df


# --------------------------------------------
# 2. CLEAN THE DATA
# --------------------------------------------
def clean_data(df):
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["amount"] = df["amount"].fillna(0)

    # Fill missing names/emails with placeholders
    df["customer_name"] = df.get("customer_name", "").fillna("Unknown").str.strip()
    df["email"] = df.get("email", "").fillna("unknown@example.com").str.strip().str.lower()

    return df


# --------------------------------------------
# 3. TRANSFORM THE DATA
# --------------------------------------------
def transform_data(df, customer_profile_file=None):
    # String manipulations
    df["customer_name"] = df["customer_name"].str.title().str.strip()
    df["city"] = df["city"].str.title()

    # Create new features
    df["transaction_month"] = df["transaction_date"].dt.to_period("M")
    df["amount"] = df["amount"].astype(float)

    # Merge with customer profile dataset if provided
    if customer_profile_file:
        profile_df = pd.read_csv(customer_profile_file)
        profile_df.columns = profile_df.columns.str.lower().str.replace(" ", "_")
        df = df.merge(profile_df, on="customer_id", how="left")

    return df


# --------------------------------------------
# 4. CHURN + SEGMENTATION
# --------------------------------------------
def apply_churn_and_segmentation(df):
    # ---- Churn Logic ----
    # Customer is churned if last purchase was > 90 days ago
    last_date = df["transaction_date"].max()
    churn_threshold = last_date - pd.Timedelta(days=90)

    cust_last_purchase = df.groupby("customer_id")["transaction_date"].max()
    churn_df = pd.DataFrame(cust_last_purchase)
    churn_df["is_churned"] = churn_df["transaction_date"] < churn_threshold

    df = df.merge(churn_df["is_churned"], on="customer_id", how="left")

    # ---- RFM Segmentation ----
    rfm = df.groupby("customer_id").agg({
        "transaction_date": lambda x: (last_date - x.max()).days,  # Recency
        "transaction_id": "count",                               # Frequency
        "amount": "sum"                                          # Monetary
    }).rename(columns={
        "transaction_date": "recency",
        "transaction_id": "frequency",
        "amount": "monetary"
    })

    # Quantile-based segmentation
    rfm["R"] = pd.qcut(rfm["recency"], 3, labels=[3, 2, 1])
    rfm["F"] = pd.qcut(rfm["frequency"], 3, labels=[1, 2, 3])
    rfm["M"] = pd.qcut(rfm["monetary"], 3, labels=[1, 2, 3])

    rfm["segment"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)

    df = df.merge(rfm[["segment"]], on="customer_id", how="left")

    return df, rfm


# --------------------------------------------
# 5. PIVOTING & RESHAPING
# --------------------------------------------
def create_pivots(df):
    monthly_revenue = df.pivot_table(
        index="transaction_month",
        columns="segment",
        values="amount",
        aggfunc="sum"
    ).fillna(0)

    melted = df.melt(
        id_vars=["customer_id", "segment"],
        value_vars=["amount"],
        var_name="metric",
        value_name="value"
    )

    return monthly_revenue, melted


# --------------------------------------------
# 6. VISUALIZATION
# --------------------------------------------
def visualize_data(df, rfm):
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df.drop_duplicates("customer_id"), x="is_churned")
    plt.title("Churn Distribution")
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df.drop_duplicates("customer_id"), x="segment")
    plt.title("Customer Segmentation (RFM)")
    plt.show()

    # Revenue trend
    rev = df.groupby("transaction_month")["amount"].sum()
    plt.figure(figsize=(10, 5))
    rev.plot(kind="line")
    plt.title("Monthly Revenue Trend")
    plt.ylabel("Revenue")
    plt.show()


# --------------------------------------------
# 7. UPLOAD TO MONGODB
# --------------------------------------------
def upload_to_mongo(df, config_file="mongo_config.toml"):
    with open(config_file, "rb") as f:
        cfg = tomli.load(f)

    host = cfg["mongo"]["host"]
    port = cfg["mongo"]["port"]
    user = cfg["mongo"]["username"]
    password = cfg["mongo"]["password"]
    dbname = cfg["mongo"]["database"]
    collection_name = cfg["mongo"]["collection"]

    uri = f"mongodb://{user}:{password}@{host}:{port}/"

    client = MongoClient(uri)
    db = client[dbname]
    collection = db[collection_name]

    records = df.to_dict("records")
    collection.insert_many(records)

    print("Upload complete.")


# ============================================
# MAIN PIPELINE
# ============================================
if __name__ == "__main__":

    df = load_data("customer_transactions.csv")

    df = clean_data(df)

    df = transform_data(df, customer_profile_file="customer_profiles.csv")  # Optional

    df, rfm = apply_churn_and_segmentation(df)

    monthly_pivot, melted_data = create_pivots(df)

    visualize_data(df, rfm)

    upload_to_mongo(df)
