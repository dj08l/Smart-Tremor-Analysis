import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Tremor Signal Simulation
# -------------------------------
def simulate_tremor(freq, fs=100, duration=5, noise=0.2):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    noise_signal = noise * np.random.randn(len(t))
    return t, signal + noise_signal

# -------------------------------
# 
# -------------------------------
def bandpass_filter(signal, fs, lowcut=3, highcut=8, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# -------------------------------
# 3. Feature Extraction
# -------------------------------
def extract_features(signal, fs):
    N = len(signal)

    # FFT
    yf = np.abs(fft(signal))
    xf = fftfreq(N, 1 / fs)

    dominant_freq = abs(xf[np.argmax(yf[:N // 2])])
    rms = np.sqrt(np.mean(signal ** 2))
    peak_to_peak = np.ptp(signal)

    return [dominant_freq, rms, peak_to_peak]

# -------------------------------
# 4. Dataset Creation
# -------------------------------
np.random.seed(42)

X = []
y = []
fs = 100

for _ in range(60):
    _, sig = simulate_tremor(freq=np.random.uniform(1, 2))
    filtered = bandpass_filter(sig, fs)
    X.append(extract_features(filtered, fs))
    y.append(0)


for _ in range(60):
    _, sig = simulate_tremor(freq=np.random.uniform(4, 6))
    filtered = bandpass_filter(sig, fs)
    X.append(extract_features(filtered, fs))
    y.append(1)

X = np.array(X)
y = np.array(y)

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -------------------------------
# 6. Train ML Model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 7. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 8. Test on New Tremor Signal
# -------------------------------
t, test_signal = simulate_tremor(freq=5)  # Parkinson-like
filtered_test = bandpass_filter(test_signal, fs)

features = np.array(extract_features(filtered_test, fs)).reshape(1, -1)
prediction = model.predict(features)

print("\nPrediction Result:")
if prediction[0] == 1:
    print("Parkinsonian Tremor Detected (Screening Risk)")
else:
    print(" Normal Tremor")

# -------------------------------
# 9. Visualization
# -------------------------------
plt.figure(figsize=(10, 4))
plt.plot(t, filtered_test)
plt.title("Filtered Tremor Signal (Time Domain)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# FFT Plot
N = len(filtered_test)
yf = np.abs(fft(filtered_test))
xf = fftfreq(N, 1 / fs)

plt.figure(figsize=(10, 4))
plt.plot(xf[:N // 2], yf[:N // 2])
plt.title("Frequency Spectrum (FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()
