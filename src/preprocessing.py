from pathlib import Path
import pandas as pd


def ensure_processed_data(raw_path, clean_path, trending_path):
    raw_path = Path(raw_path)
    clean_path = Path(clean_path)
    trending_path = Path(trending_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.lower()

    required_cols = ["product_id", "category_code", "brand", "price", "user_id"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df = df.copy()

    keep_cols = [col for col in df.columns if col in [
        "event_time", "event_type", "product_id", "category_id",
        "category_code", "brand", "price", "user_id", "user_session"
    ]]
    df = df[keep_cols]

    df["brand"] = df["brand"].fillna("Unknown")
    df["category_code"] = df["category_code"].fillna("Unknown")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["product_id", "user_id", "price"])
    df = df[df["price"] > 0]

    # friendly product name
    category_name = df["category_code"].astype(str).apply(
        lambda x: x.split(".")[-1].replace("_", " ").title()
    )
    brand_name = df["brand"].astype(str).str.title()

    df["name"] = (brand_name + " " + category_name).str.strip()

    # if brand unknown then fallback to category only
    df.loc[df["brand"].astype(str).str.lower() == "unknown", "name"] = category_name

    product_freq = df.groupby("product_id").size().reset_index(name="interaction_count")
    df = df.merge(product_freq, on="product_id", how="left")

    def make_rating(x):
        if x >= 50:
            return 5
        elif x >= 20:
            return 4
        elif x >= 10:
            return 3
        elif x >= 5:
            return 2
        return 1

    df["rating"] = df["interaction_count"].apply(make_rating)

    df["tags"] = (
        df["brand"].astype(str) + " " +
        df["category_code"].astype(str) + " " +
        df["name"].astype(str)
    )

    clean_df = df[[
        "product_id", "user_id", "name", "brand",
        "category_code", "price", "rating", "tags"
    ]].drop_duplicates()

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_path, index=False)

    trending = (
        clean_df.groupby(["product_id", "name", "brand", "category_code"], as_index=False)
        .agg(
            interaction_count=("product_id", "count"),
            avg_price=("price", "mean"),
            avg_rating=("rating", "mean")
        )
        .sort_values(by="interaction_count", ascending=False)
        .head(20)
    )

    trending["avg_price"] = trending["avg_price"].round(2)
    trending["avg_rating"] = trending["avg_rating"].round(2)
    trending.to_csv(trending_path, index=False)