from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def get_base_dir():
    return Path(__file__).resolve().parent.parent


def load_clean_data():
    base_dir = get_base_dir()
    data_path = base_dir / "data" / "clean_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"clean_data.csv not found at {data_path}")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    return df


def evaluate_recommendation_system(df, max_users=3000, top_k=5):
    print("\n🔹 Evaluating Recommendation System...")

    required_cols = {"user_id", "product_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df[["user_id", "product_id"]].dropna().copy()

    unique_users = work_df["user_id"].unique()
    if len(unique_users) > max_users:
        np.random.seed(42)
        sampled_users = np.random.choice(unique_users, size=max_users, replace=False)
        work_df = work_df[work_df["user_id"].isin(sampled_users)].copy()

    work_df["interaction"] = 1
    work_df["user_idx"] = work_df["user_id"].astype("category").cat.codes
    work_df["product_idx"] = work_df["product_id"].astype("category").cat.codes

    n_users = work_df["user_idx"].nunique()
    n_products = work_df["product_idx"].nunique()

    user_item = csr_matrix(
        (
            work_df["interaction"].values,
            (work_df["user_idx"].values, work_df["product_idx"].values)
        ),
        shape=(n_users, n_products)
    )

    similarity = cosine_similarity(user_item, dense_output=False)

    nonzero_counts = np.diff(similarity.indptr)
    avg_neighbors = float(np.mean(nonzero_counts))
    density = float(user_item.nnz / (n_users * n_products))

    metrics = {
        "sampled_users": int(n_users),
        "sampled_products": int(n_products),
        "interactions": int(user_item.nnz),
        "matrix_density": round(density, 6),
        "avg_nonzero_similarities": round(avg_neighbors, 2),
        "top_k": int(top_k),
    }

    return metrics


def evaluate_customer_segments(df):
    print("\n🔹 Evaluating Customer Segmentation...")

    required_cols = {"user_id", "price", "product_id"}
    missing = required_cols - set(df.columns)
    if missing:
        return {"segmentation_status": f"Skipped - missing columns {list(missing)}"}

    customer_df = df.groupby("user_id").agg(
        total_spent=("price", "sum"),
        avg_spent=("price", "mean"),
        total_orders=("product_id", "count"),
        unique_products=("product_id", "nunique"),
    ).reset_index()

    metrics = {
        "total_customers": int(customer_df["user_id"].nunique()),
        "avg_total_spent": round(float(customer_df["total_spent"].mean()), 2),
        "avg_order_count": round(float(customer_df["total_orders"].mean()), 2),
    }

    return metrics


def evaluate_business_metrics(df):
    print("\n🔹 Evaluating Business Metrics...")

    required_cols = {"price", "product_id", "user_id", "category_code"}
    missing = required_cols - set(df.columns)
    if missing:
        return {"business_status": f"Skipped - missing columns {list(missing)}"}

    metrics = {
        "total_revenue": round(float(df["price"].sum()), 2),
        "avg_price": round(float(df["price"].mean()), 2),
        "total_orders": int(len(df)),
        "unique_users": int(df["user_id"].nunique()),
        "unique_products": int(df["product_id"].nunique()),
        "unique_categories": int(df["category_code"].nunique()),
    }

    return metrics


def run_full_evaluation():
    base_dir = get_base_dir()
    output_path = base_dir / "data" / "evaluation_metrics.json"

    df = load_clean_data()

    rec_metrics = evaluate_recommendation_system(df)
    seg_metrics = evaluate_customer_segments(df)
    biz_metrics = evaluate_business_metrics(df)

    results = {
        "recommendation": rec_metrics,
        "segmentation": seg_metrics,
        "business_metrics": biz_metrics
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n========== EVALUATION RESULTS ==========")

    print("\nRecommendation Metrics:")
    for k, v in rec_metrics.items():
        print(f"{k}: {v}")

    print("\nSegmentation Metrics:")
    for k, v in seg_metrics.items():
        print(f"{k}: {v}")

    print("\nBusiness Metrics:")
    for k, v in biz_metrics.items():
        print(f"{k}: {v}")

    print("\n✅ Evaluation completed successfully.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    run_full_evaluation()