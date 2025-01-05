import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset (already cleaned and preprocessed)
@st.cache
def load_data():
    file_path = '/mnt/data/used_cars.csv'
    df = pd.read_csv(file_path)

    # Data Cleaning
    df['price'] = df['price'].str.replace(r'[\$,]', '', regex=True).astype(float)
    df['milage'] = df['milage'].str.replace(r'[\, mi.]', '', regex=True).astype(float)
    df['engine_capacity'] = df['engine'].str.extract(r'(\d+\.\d+)').astype(float)
    df['accident'] = df['accident'].apply(lambda x: 1 if 'accident' in str(x).lower() else 0)
    df['fuel_type'] = df['fuel_type'].fillna('Unknown')
    df['clean_title'] = df['clean_title'].fillna('No')
    df['clean_title'] = df['clean_title'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    selected_features = ['brand', 'model', 'model_year', 'milage', 'fuel_type', 'transmission',
                         'engine_capacity', 'accident', 'clean_title']
    df_cleaned = df[selected_features + ['price']]
    return df_cleaned

data = load_data()

# Encode categorical variables
def preprocess_data(df):
    df = df.copy()
    encoders = {}

    for col in ['brand', 'model', 'fuel_type', 'transmission']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

# Split data into training and test sets
def split_data(df):
    X = df.drop(columns=['price'])
    y = df['price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# Load and preprocess data
data, encoders = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(data)
model = train_model(X_train, y_train)

# Save the model
joblib.dump(model, 'car_price_model.pkl')

# Streamlit App
st.title("Used Car Price Prediction App")

# User Inputs
st.sidebar.header("Input Car Details")
brand = st.sidebar.selectbox("Brand", encoders['brand'].classes_)
model = st.sidebar.selectbox("Model", encoders['model'].classes_)
model_year = st.sidebar.slider("Model Year", int(data['model_year'].min()), int(data['model_year'].max()), 2020)
milage = st.sidebar.number_input("Mileage (in miles)", min_value=0, step=1000)
fuel_type = st.sidebar.selectbox("Fuel Type", encoders['fuel_type'].classes_)
transmission = st.sidebar.selectbox("Transmission", encoders['transmission'].classes_)
engine_capacity = st.sidebar.number_input("Engine Capacity (in liters)", min_value=0.0, step=0.1)
accident = st.sidebar.selectbox("Accident History", ['None', 'At least 1 accident'])
clean_title = st.sidebar.selectbox("Clean Title", ['Yes', 'No'])

# Transform inputs for prediction
input_data = {
    'brand': encoders['brand'].transform([brand])[0],
    'model': encoders['model'].transform([model])[0],
    'model_year': model_year,
    'milage': milage,
    'fuel_type': encoders['fuel_type'].transform([fuel_type])[0],
    'transmission': encoders['transmission'].transform([transmission])[0],
    'engine_capacity': engine_capacity,
    'accident': 1 if accident == 'At least 1 accident' else 0,
    'clean_title': 1 if clean_title == 'Yes' else 0
}
input_df = pd.DataFrame([input_data])

# Predict price
predicted_price = model.predict(input_df)[0]
st.write(f"### Predicted Price: ${predicted_price:,.2f}")

# Visualization
st.header("Data Insights")

# Line Graph: Average Price by Year
avg_price_by_year = data.groupby('model_year')['price'].mean()
st.line_chart(avg_price_by_year)

# Bar Chart: Average Price by Brand
avg_price_by_brand = data.groupby('brand')['price'].mean().sort_values(ascending=False)
st.bar_chart(avg_price_by_brand)

# Heatmap: Correlation Matrix
st.write("### Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
st.pyplot(plt)

# Future Price Analysis
st.header("Future Price Analysis")
price_decay_rate = 0.15  # Assuming 15% yearly depreciation
future_prices = {f"Year {i+1}": predicted_price * ((1 - price_decay_rate) ** i) for i in range(5)}
st.write(pd.DataFrame(future_prices, index=["Price"]).T)
