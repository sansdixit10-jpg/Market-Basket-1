import streamlit as st
import pandas as pd
from apriori_model import generate_rules

st.title("🛒 Market Basket Recommendation System")

df = pd.read_csv("cleaned_data.csv")

basket = (
    df.groupby(['InvoiceNo', 'Description'])['Quantity']
    .sum()
    .unstack()
    .fillna(0)
)

basket = (basket > 0).astype(bool)

rules = generate_rules(basket)

product_list = sorted(df['Description'].unique())
selected_product = st.selectbox("Select a Product", product_list)

st.subheader("🎯 Recommended Products")

if not rules.empty:
    recommendations = rules[
        rules['antecedents'].apply(lambda x: selected_product in x)
    ].sort_values(by='lift', ascending=False)

    for _, row in recommendations.iterrows():
        st.write("👉", list(row['consequents'])[0])
else:
    st.warning("No rules generated.")
