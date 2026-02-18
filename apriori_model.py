from mlxtend.frequent_patterns import apriori, association_rules

def generate_rules(basket):

    frequent_itemsets = apriori(
        basket,
        min_support=0.01,
        use_colnames=True,
        max_len=2
    )

    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=1
    )

    return rules
