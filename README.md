import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("🔧 Predictive Maintenance for Machines")
st.write("This app predicts if a machine needs maintenance using sensor data.")

uploaded_file = st.file_uploader("Upload sensor data CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(data.head())

    if 'condition' not in data.columns:
        st.error("CSV must have a 'condition' column with labels (e.g., 0 = normal, 1 = failure).")
    else:
        X = data.drop('condition', axis=1)
        y = data['condition']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        st.subheader("✅ Results")
        st.write(f"Model Accuracy: **{acc * 100:.2f}%**")

        st.subheader("📊 Predict Maintenance")
        sample_input = st.text_input("Enter sensor values (comma-separated):")

        if sample_input:
            try:
                values = [float(val) for val in sample_input.split(",")]
                pred = model.predict([values])
                st.success("Prediction: 🟢 Normal" if pred[0] == 0 else "Prediction: 🔴 Needs Maintenance")
            except:
                st.error("Invalid input. Make sure you enter the correct number of numeric values.")
