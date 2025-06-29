import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

# Sidebar Inputs
st.sidebar.title("Enter House Features")

MedInc = st.sidebar.slider("MedInc (Median Income)", 0.5, 15.0, 3.0)
AveRooms = st.sidebar.slider("AveRooms", 1.0, 10.0, 5.0)
AveBedrms = st.sidebar.slider("AveBedrms", 0.5, 5.0, 1.0)
Population = st.sidebar.slider("Population", 100.0, 5000.0, 1000.0)
AveOccup = st.sidebar.slider("AveOccup", 1.0, 10.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 36.0)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -120.0)
HouseAge = st.sidebar.slider("HouseAge", 1, 50, 20)

# Prepare the input
input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]],
                          columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])

# Data Preprocessing
X = df.drop("Target", axis=1)
y = df["Target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

# Display
st.title("🏡 California House Price Prediction")
st.write("### Predicted House Value (in 100,000s of $):")
st.success(f"${prediction[0]*100000:,.2f}")
