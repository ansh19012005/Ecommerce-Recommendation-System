import pandas as pd


def collaborative_filtering(user_id, df, top_n=8):
    if df.empty:
        return []

    data = df.copy()

    # if user not found
    if user_id not in data["user_id"].values:
        return []

    # items already interacted by user
    user_items = set(data.loc[data["user_id"] == user_id, "product_id"].tolist())

    # find similar users by overlap
    target_categories = set(
        data.loc[data["user_id"] == user_id, "category_code"].astype(str).tolist()
    )

    other_users = data[data["user_id"] != user_id].copy()
    if other_users.empty:
        return []

    similarity_scores = []

    for other_id, group in other_users.groupby("user_id"):
        other_categories = set(group["category_code"].astype(str).tolist())
        overlap = len(target_categories.intersection(other_categories))
        if overlap > 0:
            similarity_scores.append((other_id, overlap))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_user_ids = [u for u, _ in similarity_scores[:20]]

    if not similar_user_ids:
        return []

    candidates = data[data["user_id"].isin(similar_user_ids)].copy()
    candidates = candidates[~candidates["product_id"].isin(user_items)]

    if candidates.empty:
        return []

    recs = (
        candidates.groupby(
            ["product_id", "Name", "brand", "category_code"], as_index=False
        )
        .agg(
            score=("Rating", "mean"),
            price=("price", "mean"),
            frequency=("product_id", "count")
        )
        .sort_values(by=["frequency", "score"], ascending=False)
        .head(top_n)
    )

    recs["price"] = recs["price"].round(2)
    recs["score"] = recs["score"].round(2)

    return recs.to_dict(orient="records")