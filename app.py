import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import os
import joblib


if not os.path.exists("price_model.pkl"):
    import train_model 

model=joblib.load("price_model.pkl")

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="RealtyAI",
    layout="wide"
)

# ---------------- BACKGROUND STYLE ----------------

page_bg = """
<style>

[data-testid="stAppViewContainer"]{
background-image: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
url("https://images.unsplash.com/photo-1600585154340-be6161a56a0c");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}

[data-testid="stSidebar"]{
background-color: rgba(0,0,0,0.7);
}

h1, h2, h3, h4, p, label{
color: white;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

df = pd.read_csv("data/housing.csv")

# Data preprocessing for UI
df_clean = df[["location","size","total_sqft","bath","balcony","price"]]
df_clean = df_clean.dropna()

df_clean["total_sqft"] = pd.to_numeric(df_clean["total_sqft"], errors="coerce")
df_clean["bedrooms"] = df_clean["size"].str.extract("(\d+)")

df_clean = df_clean.dropna()

# Encode location
le = LabelEncoder()
df_clean["location_encoded"] = le.fit_transform(df_clean["location"])

# Load model
model = pickle.load("price_model.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.title("RealtyAI")

page = st.sidebar.radio(
    "Navigation",
    ["Home","Price Prediction","Budget Recommendation","Market Analysis","Dataset"]
)

# ---------------- HOME ----------------

if page == "Home":

    st.title("RealtyAI - Smart Real Estate Insight Platform")

    st.write("""
    RealtyAI is an AI powered platform that analyzes real estate data
    and predicts housing prices using machine learning.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Properties", len(df_clean))
    col2.metric("Average Price (Lakhs)", round(df_clean['price'].mean(),2))
    col3.metric("Locations", df_clean["location"].nunique())

    st.image(
        "https://images.unsplash.com/photo-1560518883-ce09059eeffa",
        width=400
    )

# ---------------- PRICE PREDICTION ----------------

elif page == "Price Prediction":

    st.title("Property Price Prediction")

    locations = df_clean["location"].unique()

    col1, col2 = st.columns(2)

    with col1:

        location_selected = st.selectbox("Select Location", locations)

        sqft = st.number_input("Total Sqft",300,10000)

        bedrooms = st.number_input("Bedrooms",1,10)

    with col2:

        bath = st.number_input("Bathrooms",1,10)

        balcony = st.number_input("Balconies",0,5)

    if st.button("Predict Price"):

        location_code = le.transform([location_selected])[0]

        features = np.array([[location_code, sqft, bath, balcony, bedrooms]])

        prediction = model.predict(features)

        st.success(f"Estimated Price: ₹ {prediction[0]:,.2f} Lakhs")

# ---------------- BUDGET RECOMMENDATION ----------------

elif page == "Budget Recommendation":

    st.title("Budget Based Location Recommendation")

    budget = st.number_input("Enter Your Budget (Lakhs)",10,500)

    avg_prices = df_clean.groupby("location")["price"].mean()

    affordable = avg_prices[avg_prices <= budget].sort_values()

    if len(affordable) > 0:

        st.write("Locations within your budget:")

        st.write(affordable.head(10))

    else:

        st.write("No locations found within this budget.")

#  MARKET ANALYSIS 

elif page == "Market Analysis":

    st.title("Real Estate Market Insights")

    fig1 = px.histogram(
        df_clean,
        x="price",
        nbins=40,
        title="Price Distribution"
    )

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        df_clean,
        x="total_sqft",
        y="price",
        title="Price vs Area"
    )

    st.plotly_chart(fig2, use_container_width=True)

    avg_price = df_clean.groupby("location")["price"].mean().sort_values(ascending=False).head(10)

    fig3 = px.bar(
        avg_price,
        title="Top 10 Expensive Locations"
    )

    st.plotly_chart(fig3, use_container_width=True)

# DATASET

elif page == "Dataset":

    st.title("Housing Dataset")

    st.dataframe(df_clean)

    st.write("Dataset Shape:", df_clean.shape)


