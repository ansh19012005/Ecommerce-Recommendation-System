import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def prepare_basket_data(df: pd.DataFrame, basket_col="user_id", item_col="category_code") -> pd.DataFrame:
    required_cols = [basket_col, item_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    basket = (
        df[[basket_col, item_col]]
        .dropna()
        .drop_duplicates()
        .assign(value=1)
        .pivot_table(index=basket_col, columns=item_col, values="value", fill_value=0)
    )

    basket = basket.astype(bool)
    return basket


def run_market_basket_analysis(
    input_csv="data/clean_data.csv",
    output_csv="data/market_basket_rules.csv",
    min_support=0.01,
    metric="lift",
    min_threshold=1.0
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    basket = prepare_basket_data(df, basket_col="user_id", item_col="category_code")

    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        print("No frequent itemsets found. Try lowering min_support.")
        return None

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    if rules.empty:
        print("No association rules generated. Try lowering thresholds.")
        return None

    rules = rules.copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

    selected_cols = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]

    rules = rules[selected_cols].sort_values(by=["lift", "confidence"], ascending=False)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rules.to_csv(output_csv, index=False)

    print("Market Basket Analysis completed successfully.")
    print(f"Rules saved to: {output_csv}")
    print(f"Total rules generated: {len(rules)}")

    print("\nTop 10 Rules:")
    print(rules.head(10).to_string(index=False))

    return rules


if __name__ == "__main__":
    run_market_basket_analysis()