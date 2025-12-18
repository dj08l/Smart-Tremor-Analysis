# -*- coding: utf-8 -*-

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------
# Signal Simulation
# -------------------------------
def simulate_tremor(freq, fs=100, duration=5, noise=0.2):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    noise_signal = noise * np.random.randn(len(t))
    return t, signal + noise_signal

# -------------------------------
# Band-pass Filter (3-8 Hz)
# -------------------------------
def bandpass_filter(signal, fs, low=3, high=8, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_features(signal, fs):
    N = len(signal)
    yf = np.abs(fft(signal))
    xf = fftfreq(N, 1/fs)

    dominant_freq = abs(xf[np.argmax(yf[:N//2])])
    rms = np.sqrt(np.mean(signal ** 2))
    peak_to_peak = np.ptp(signal)

    return [dominant_freq, rms, peak_to_peak]

# -------------------------------
# Train ML Model
# -------------------------------
def train_model():
    X, y = [], []
    fs = 100

    for _ in range(60):
        _, sig = simulate_tremor(np.random.uniform(1, 2))
        sig = bandpass_filter(sig, fs)
        X.append(extract_features(sig, fs))
        y.append(0)

    for _ in range(60):
        _, sig = simulate_tremor(np.random.uniform(4, 6))
        sig = bandpass_filter(sig, fs)
        X.append(extract_features(sig, fs))
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Smart Tremor Analysis for Parkinson Screening")
st.write("Software-based biomedical signal analysis system")

model = train_model()

freq = st.slider("Select Tremor Frequency (Hz)", 1.0, 7.0, 5.0)
fs = 100

t, signal = simulate_tremor(freq)
filtered = bandpass_filter(signal, fs)

features = np.array(extract_features(filtered, fs)).reshape(1, -1)
prediction = model.predict(features)

# Result
st.subheader("Screening Result")
if prediction[0] == 1:
    st.error("Parkinsonian Tremor Detected (Screening Risk)")
else:
    st.success("Normal Tremor Detected")

# Time domain plot
st.subheader("Filtered Tremor Signal (Time Domain)")
fig1, ax1 = plt.subplots()
ax1.plot(t, filtered)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
st.pyplot(fig1)

# Frequency domain plot
st.subheader("Frequency Spectrum (FFT)")
N = len(filtered)
yf = np.abs(fft(filtered))
xf = fftfreq(N, 1/fs)

fig2, ax2 = plt.subplots()
ax2.plot(xf[:N//2], yf[:N//2])
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
st.pyplot(fig2)

st.caption("For screening and monitoring only. Not a medical diagnostic tool.")
