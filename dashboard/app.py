# File: /dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Title
st.title("Predictive Monitoring System Dashboard")

# Simulate live data
np.random.seed(42)
gas = np.random.normal(100, 10, 1000)
temperature = np.random.normal(35, 3, 1000)
vibration = np.random.normal(0.25, 0.05, 1000)

# Inject sample anomalies
gas[500:510] += 40

# Create DataFrame
data = pd.DataFrame({
    'gas': gas,
    'temperature': temperature,
    'vibration': vibration
})

# Apply Isolation Forest
model = IsolationForest(contamination=0.02, random_state=42)
data['anomaly_score'] = model.fit_predict(data[['gas', 'temperature', 'vibration']])
data['anomaly'] = data['anomaly_score'] == -1

# Display metrics
st.metric("Total Records", len(data))
st.metric("Detected Anomalies", data['anomaly'].sum())

# Chart: Gas + Anomalies
st.subheader("Gas Readings with Anomalies")
st.line_chart(data['gas'])
anomaly_index = data[data['anomaly']].index
st.write("Anomaly Indices:", list(anomaly_index))

# Optional CSV Export
csv = data.to_csv(index=False).encode('utf-8')
st.download_button("Download Sensor Data as CSV", csv, "sensor_data.csv", "text/csv")
