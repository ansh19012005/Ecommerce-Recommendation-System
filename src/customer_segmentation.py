import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["user_id", "product_id", "price", "rating"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    customer_df = (
        df.groupby("user_id")
        .agg(
            total_interactions=("product_id", "count"),
            unique_products=("product_id", "nunique"),
            avg_price=("price", "mean"),
            avg_rating=("rating", "mean"),
        )
        .reset_index()
    )

    customer_df["avg_price"] = customer_df["avg_price"].fillna(0)
    customer_df["avg_rating"] = customer_df["avg_rating"].fillna(0)

    return customer_df


def assign_segment_names(segmented_df: pd.DataFrame) -> pd.DataFrame:
    cluster_summary = (
        segmented_df.groupby("cluster")
        .agg(
            total_interactions=("total_interactions", "mean"),
            avg_price=("avg_price", "mean"),
            avg_rating=("avg_rating", "mean"),
            unique_products=("unique_products", "mean"),
        )
        .reset_index()
    )

    cluster_summary = cluster_summary.sort_values(
        by=["total_interactions", "avg_price", "avg_rating"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    default_names = [
        "Premium Customers",
        "Frequent Buyers",
        "Regular Customers",
        "Low Engagement Users"
    ]

    cluster_name_map = {}
    for i, row in cluster_summary.iterrows():
        cluster_id = row["cluster"]
        if i < len(default_names):
            cluster_name_map[cluster_id] = default_names[i]
        else:
            cluster_name_map[cluster_id] = f"Customer Segment {i+1}"

    segmented_df["segment_name"] = segmented_df["cluster"].map(cluster_name_map)
    return segmented_df


def run_customer_segmentation(
    input_csv="data/clean_data.csv",
    output_csv="data/customer_segments.csv",
    model_path="models/kmeans_model.pkl",
    scaler_path="models/scaler.pkl",
    n_clusters=4,
    random_state=42
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    customer_df = build_customer_features(df)

    feature_cols = ["total_interactions", "unique_products", "avg_price", "avg_rating"]
    X = customer_df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    customer_df["cluster"] = kmeans.fit_predict(X_scaled)

    customer_df = assign_segment_names(customer_df)

    score = silhouette_score(X_scaled, customer_df["cluster"])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    customer_df.to_csv(output_csv, index=False)
    joblib.dump(kmeans, model_path)
    joblib.dump(scaler, scaler_path)

    print("Customer segmentation completed successfully.")
    print(f"Segments saved to: {output_csv}")
    print(f"KMeans model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Silhouette Score: {score:.4f}")

    print("\nSample segment distribution:")
    print(customer_df["segment_name"].value_counts())

    return customer_df, score


if __name__ == "__main__":
    run_customer_segmentation()