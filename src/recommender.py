def get_product_names(df):
    if "name" not in df.columns:
        df["name"] = df["product_id"].astype(str)

    names = sorted(df["name"].dropna().astype(str).unique().tolist())
    return names


def content_based_recommendations(df, product_name, top_n=8):
    if "name" not in df.columns:
        df["name"] = df["product_id"].astype(str)

    filtered = df[df["name"] == product_name]

    if filtered.empty:
        return []

    category = filtered.iloc[0]["category_code"]

    similar = df[df["category_code"] == category].copy()
    similar = similar.drop_duplicates(subset=["product_id"])

    results = []
    for _, row in similar.head(top_n).iterrows():
        results.append({
            "product_id": row.get("product_id"),
            "name": row.get("name"),
            "brand": row.get("brand"),
            "category_code": row.get("category_code"),
            "price": round(float(row.get("price", 0)), 2),
            "similarity_score": 1.0
        })

    return results


def get_trending_products(df, limit=8):
    df = df.copy()

    if "name" not in df.columns:
        df["name"] = df["product_id"].astype(str)

    # Use avg_price instead of price
    if "avg_price" not in df.columns:
        df["avg_price"] = 0

    trending = (
        df.sort_values(by="avg_price", ascending=False)
        .head(limit)
    )

    return trending.to_dict(orient="records")