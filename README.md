# Predictive Monitoring System for Underground Mines

This project is a prototype for an AI-powered predictive monitoring system designed to enhance safety in underground mining environments by forecasting hazardous conditions and detecting anomalies in real time.

![Predictive Monitoring System for Underground Mines](https://github.com/user-attachments/assets/257661a0-79ff-456a-b6ab-b9dc541855a0)


---

## 📂 Project Structure

```
├── notebooks/
│   └── predict_anomalies.ipynb     # Jupyter Notebook: simulate data, run LSTM & Isolation Forest
├── models/
│   └── lstm_model.py               # LSTM forecasting model class
├── dashboard/
│   └── app.py                      # Streamlit dashboard for sensor data visualization
├── dashboard_flask/
│   ├── flask_app.py                # Flask-based dashboard backend
│   ├── templates/
│   │   └── index.html              # Frontend HTML UI
│   └── static/css/style.css        # Custom CSS for dashboard styling
├── README.md                       # This file
```

---

## 📊 Features

- Simulates real-time sensor data: Gas, Temperature, Vibration
- Detects anomalies using Isolation Forest
- Forecasts future gas trends using LSTM
- Two dashboard options:
  - Streamlit UI (fast prototype)
  - Flask UI (custom HTML/CSS styled frontend)

---

## ⚙️ Installation & Setup

```bash
# Clone the repo
https://github.com/your-username/mining-safety-monitor
cd mining-safety-monitor

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# Install dependencies
pip install flask numpy pandas scikit-learn matplotlib seaborn tensorflow streamlit jupyter
```

---

## 🚀 Running the Notebook

Open the notebook with Jupyter or VS Code:
```bash
jupyter notebook notebooks/predict_anomalies.ipynb
```

---

## 🖥️ Running the Streamlit Dashboard
```bash
cd dashboard
streamlit run app.py
```

---

## 🖥️ Running the Flask Dashboard
```bash
cd dashboard_flask
python flask_app.py
```
Then open `http://localhost:5000` in your browser.

---

## 📦 Dependencies

This project relies on:
- `numpy`, `pandas` – data generation and transformation
- `scikit-learn` – anomaly detection (Isolation Forest)
- `tensorflow` – LSTM model (includes `keras` layers and models)
- `matplotlib`, `seaborn` – visual plots
- `streamlit` – rapid dashboard prototyping
- `flask` – HTML-based custom dashboard
- `jupyter` – running the notebook

Install all at once:
```bash
pip install flask numpy pandas scikit-learn matplotlib seaborn tensorflow streamlit jupyter
```

---

## 📩 Contact
**Author:** HABIMANA Jean de Dieu  
**Email:** jeandedh@andrew.cmu.edu
