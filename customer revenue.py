# ============================================
# Customer Revenue & Churn Intelligence Functions
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

    # Fix dates
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    # Amount handling
    if "amount" in df.columns:
        df["amount"] = df["amount"].fillna(0).astype(float)

    # Name handling
    df["customer_name"] = df.get("customer_name", pd.Series(["Unknown"] * len(df))).fillna("Unknown")

    # Email handling
    df["email"] = df.get("email", pd.Series(["unknown@example.com"] * len(df))).fillna("unknown@example.com")

    return df


# --------------------------------------------
# 3. TRANSFORM THE DATA
# --------------------------------------------
def transform_data(df, customer_profile_file=None):
    # String cleanup
    df["customer_name"] = df["customer_name"].astype(str).str.title().str.strip()

    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.title()

    # Create features
    df["transaction_month"] = df["transaction_date"].dt.to_period("M")

    # Optional merging
    if customer_profile_file:
        profile_df = pd.read_csv(customer_profile_file)
        profile_df.columns = profile_df.columns.str.lower().str.replace(" ", "_")
        df = df.merge(profile_df, on="customer_id", how="left")

    return df


# --------------------------------------------
# 4. CHURN + SEGMENTATION
# --------------------------------------------
def apply_churn_and_segmentation(df):
    last_date = df["transaction_date"].max()
    churn_threshold = last_date - pd.Timedelta(days=90)

    # Churn logic
    cust_last_purchase = df.groupby("customer_id")["transaction_date"].max()
    churn_df = pd.DataFrame(cust_last_purchase)
    churn_df["is_churned"] = churn_df["transaction_date"] < churn_threshold

    df = df.merge(churn_df["is_churned"], on="customer_id", how="left")

    # RFM logic
    rfm = df.groupby("customer_id").agg({
        "transaction_date": lambda x: (last_date - x.max()).days,
        "transaction_id": "count" if "transaction_id" in df.columns else "size",
        "amount": "sum"
    }).rename(columns={
        "transaction_date": "recency",
        "transaction_id": "frequency",
        "amount": "monetary"
    })

    # Ranking safe mode
    rfm["R"] = pd.qcut(rfm["recency"].rank(method="first"), 3, labels=[3, 2, 1])
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 3, labels=[1, 2, 3])
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 3, labels=[1, 2, 3])

    rfm["segment"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)

    df = df.merge(rfm[["segment"]], on="customer_id", how="left")

    return df, rfm


# --------------------------------------------
# 5. PIVOTING & RESHAPING
# --------------------------------------------
def create_pivots(df):
    monthly_revenue = df.pivot_table(
        index="transaction_month",
        values="amount",
        aggfunc="sum"
    )

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
    sns.set_style("whitegrid")

    # Churn chart
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df.drop_duplicates("customer_id"), x="is_churned")
    plt.title("Churn Distribution")
    plt.show()

    # Segmentation chart
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df.drop_duplicates("customer_id"), x="segment")
    plt.title("Customer Segmentation (RFM)")
    plt.xticks(rotation=45)
    plt.show()

    # Monthly revenue
    rev = df.groupby("transaction_month")["amount"].sum()
    plt.figure(figsize=(10, 5))
    rev.plot(kind="line")
    plt.title("Monthly Revenue Trend")
    plt.ylabel("Revenue")
    plt.show()


# --------------------------------------------
# 7. MAIN EXECUTION PIPELINE (your file added)
# --------------------------------------------
if __name__ == "__main__":
    file_path = "/mnt/data/project1-retail-raw-dataset.csv.xlsx"

    print("Loading data...")
    df = load_data(file_path)

    print("\nCleaning data...")
    df = clean_data(df)

    print("\nTransforming data...")
    df = transform_data(df)

    print("\nApplying churn and segmentation...")
    df, rfm = apply_churn_and_segmentation(df)

    print("\nCreating pivots...")
    monthly_revenue, melted = create_pivots(df)

    print("\nVisualizing results...")
    visualize_data(df, rfm)

    print("\nPipeline Complete!")
