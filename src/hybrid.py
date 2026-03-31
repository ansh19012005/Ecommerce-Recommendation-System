def hybrid_recommendation(content_recs, collab_recs, top_n=8):
    merged = {}
    order = 0

    for item in content_recs:
        pid = item["product_id"]
        merged[pid] = {
            "product_id": item.get("product_id"),
            "Name": item.get("Name", ""),
            "brand": item.get("brand", ""),
            "category_code": item.get("category_code", ""),
            "price": round(float(item.get("price", 0)), 2),
            "score": float(item.get("similarity_score", 0)) + 1.0,
            "order": order,
        }
        order += 1

    for item in collab_recs:
        pid = item["product_id"]
        if pid in merged:
            merged[pid]["score"] += float(item.get("score", 0))
        else:
            merged[pid] = {
                "product_id": item.get("product_id"),
                "Name": item.get("Name", ""),
                "brand": item.get("brand", ""),
                "category_code": item.get("category_code", ""),
                "price": round(float(item.get("price", 0)), 2),
                "score": float(item.get("score", 0)),
                "order": order,
            }
            order += 1

    results = list(merged.values())
    results.sort(key=lambda x: (-x["score"], x["order"]))
    return results[:top_n]