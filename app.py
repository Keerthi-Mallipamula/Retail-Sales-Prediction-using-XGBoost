import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Walmart Forecasting", layout="wide")

st.title("📊 Walmart Demand Forecasting Dashboard")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("final_model.pkl", "rb"))

model = load_model()

# -------------------------------
# Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Walmart Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Fix Date Format
    # -------------------------------
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(['Store', 'Date'])

    st.success("✅ Data Loaded Successfully")

    # -------------------------------
    # Sidebar Filters
    # -------------------------------
    st.sidebar.header("🔎 Filters")

    store = st.sidebar.selectbox("Select Store", sorted(df['Store'].unique()))

    filtered_df = df[df['Store'] == store].copy()

    # -------------------------------
    # Show Data
    # -------------------------------
    if st.checkbox("Show Data"):
        st.write(filtered_df.tail())

    # -------------------------------
    # KPI Metrics
    # -------------------------------
    st.subheader("📌 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Sales", f"${filtered_df['Weekly_Sales'].mean():,.0f}")
    col2.metric("Max Sales", f"${filtered_df['Weekly_Sales'].max():,.0f}")
    col3.metric("Min Sales", f"${filtered_df['Weekly_Sales'].min():,.0f}")

    # -------------------------------
    # Sales Trend
    # -------------------------------
    st.subheader("📈 Sales Trend")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(filtered_df['Date'], filtered_df['Weekly_Sales'])

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.xticks(rotation=45)
    ax.set_title(f"Store {store} Sales Trend")

    st.pyplot(fig)

    # -------------------------------
    # Feature Engineering
    # -------------------------------
    filtered_df['year'] = filtered_df['Date'].dt.year
    filtered_df['month'] = filtered_df['Date'].dt.month
    filtered_df['week'] = filtered_df['Date'].dt.isocalendar().week

    filtered_df['lag_1'] = filtered_df['Weekly_Sales'].shift(1)
    filtered_df['lag_2'] = filtered_df['Weekly_Sales'].shift(2)
    filtered_df['lag_4'] = filtered_df['Weekly_Sales'].shift(4)

    filtered_df['rolling_mean_4'] = filtered_df['Weekly_Sales'].shift(1).rolling(4).mean()
    filtered_df['rolling_std_4'] = filtered_df['Weekly_Sales'].shift(1).rolling(4).std()

    filtered_df = filtered_df.dropna()

    # -------------------------------
    # Prediction Section
    # -------------------------------
    st.subheader("🔮 Predict Next Week Sales")

    if st.button("Predict"):

        latest = filtered_df.iloc[-1]

        input_data = pd.DataFrame([{
            'Store': latest['Store'],
            'Holiday_Flag': latest['Holiday_Flag'],
            'Temperature': latest['Temperature'],
            'Fuel_Price': latest['Fuel_Price'],
            'CPI': latest['CPI'],
            'Unemployment': latest['Unemployment'],
            'year': latest['year'],
            'month': latest['month'],
            'week': latest['week'],
            'lag_1': latest['lag_1'],
            'lag_2': latest['lag_2'],
            'lag_4': latest['lag_4'],
            'rolling_mean_4': latest['rolling_mean_4'],
            'rolling_std_4': latest['rolling_std_4']
        }])

        prediction = model.predict(input_data)[0]

        st.metric("💰 Predicted Weekly Sales", f"${prediction:,.2f}")

        # -------------------------------
        # Trend Insight
        # -------------------------------
        last_sales = latest['Weekly_Sales']

        if prediction > last_sales:
            st.info("📊 Demand is expected to increase compared to last week")
        else:
            st.info("📊 Demand is expected to decrease compared to last week")

        # -------------------------------
        # Business Recommendations
        # -------------------------------
        st.subheader("💡 Business Recommendation")

        if prediction > 1800000:
            st.warning("""
            🚀 Very High Demand Expected  
            • Increase inventory immediately  
            • Add extra staff  
            • Ensure supply chain readiness  
            """)

        elif prediction > 1400000:
            st.info("""
            📈 Moderate High Demand  
            • Slightly increase stock  
            • Monitor fast-moving items  
            """)

        elif prediction < 900000:
            st.error("""
            📉 Low Demand Expected  
            • Run discounts/promotions  
            • Reduce inventory levels  
            """)

        else:
            st.success("""
            ✅ Stable Demand  
            • Maintain current operations  
            """)

    # -------------------------------
    # Prediction vs Actual
    # -------------------------------
    st.subheader("📊 Actual vs Predicted")

    features = ['Store','Holiday_Flag','Temperature','Fuel_Price',
                'CPI','Unemployment','year','month','week',
                'lag_1','lag_2','lag_4','rolling_mean_4','rolling_std_4']

    preds = model.predict(filtered_df[features])

    fig2, ax2 = plt.subplots(figsize=(12,5))

    ax2.plot(filtered_df['Date'], filtered_df['Weekly_Sales'], label="Actual")
    ax2.plot(filtered_df['Date'], preds, label="Predicted")

    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.xticks(rotation=45)

    ax2.legend()
    ax2.set_title("Actual vs Predicted Sales")

    st.pyplot(fig2)

else:
    st.info("📂 Please upload dataset to continue")