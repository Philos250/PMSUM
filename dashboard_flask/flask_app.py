# File: /dashboard_flask/flask_app.py

from flask import Flask, render_template, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lstm_model import LSTMForecaster
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route('/')
def index():
    np.random.seed(42)
    n = 500
    gas = np.random.normal(100, 10, n)
    temperature = np.random.normal(35, 3, n)
    vibration = np.random.normal(0.25, 0.05, n)
    gas[150:160] += 30

    data = pd.DataFrame({
        'gas': gas,
        'temperature': temperature,
        'vibration': vibration
    })

    model = IsolationForest(contamination=0.02, random_state=42)
    data['anomaly_score'] = model.fit_predict(data[['gas', 'temperature', 'vibration']])
    data['anomaly'] = data['anomaly_score'] == -1
    data['index'] = data.index

    # Calculate averages for dashboard metrics
    avg_gas = round(data['gas'].mean(), 2)
    avg_temp = round(data['temperature'].mean(), 2)
    avg_vib = round(data['vibration'].mean(), 2)

    # Forecast gas levels using LSTM
    forecaster = LSTMForecaster(sequence_length=20)
    forecaster.fit(data['gas'].values)
    forecast_gas = forecaster.forecast(data['gas'].values, n_steps=20).flatten().tolist()

    return render_template(
        "index.html",
        records=data.to_dict(orient='records'),
        avg_gas=avg_gas,
        avg_temp=avg_temp,
        avg_vib=avg_vib,
        gas_list=data['gas'].tolist(),
        temp_list=data['temperature'].tolist(),
        vib_list=data['vibration'].tolist(),
        forecast_gas=forecast_gas
    )

@app.route('/export-excel')
def export_excel():
    np.random.seed(42)
    n = 500
    gas = np.random.normal(100, 10, n)
    temperature = np.random.normal(35, 3, n)
    vibration = np.random.normal(0.25, 0.05, n)

    data = pd.DataFrame({
        'Gas Level': gas,
        'Temperature': temperature,
        'Vibration': vibration
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='SensorData')
    output.seek(0)
    return send_file(output, download_name="sensor_report.xlsx", as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    # app.run(debug=True)
