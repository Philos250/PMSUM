# File: /models/lstm_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.preprocessing import MinMaxScaler

class LSTMForecaster:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = self._build_model()
        self.scaler = MinMaxScaler()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(self.sequence_length, 1)))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self, series):
        scaled_series = self.scaler.fit_transform(series.reshape(-1, 1))
        X, y = [], []
        for i in range(len(scaled_series) - self.sequence_length):
            X.append(scaled_series[i:i+self.sequence_length])
            y.append(scaled_series[i+self.sequence_length])
        return np.array(X), np.array(y)

    def fit(self, series, epochs=10, batch_size=16):
        X, y = self.prepare_data(series)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def forecast(self, last_sequence, n_steps=50):
        forecast = []
        input_seq = self.scaler.transform(last_sequence.reshape(-1, 1))[-self.sequence_length:]
        input_seq = input_seq.reshape(1, self.sequence_length, 1)
        for _ in range(n_steps):
            pred = self.model.predict(input_seq, verbose=0)[0][0]
            forecast.append(pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
        return self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
