import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# 1. Define the path to the first recording in training-a
file_path = os.path.join("data", "training-a", "a0001.wav")

# 2. Load the raw PCG audio
fs, raw_audio = wavfile.read("C:/Users/Dell/Desktop/digitaltwin/Data/training-a/a0001.wav")

print(f"Successfully loaded {file_path}")
print(f"Sampling Rate: {fs} Hz")
print(f"Duration: {len(raw_audio) / fs:.2f} seconds")

# 3. Apply the Zero-Phase Butterworth Bandpass Filter (25-400 Hz)
def apply_bandpass_filter(data, fs, lowcut=25.0, highcut=400.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

clean_audio = apply_bandpass_filter(raw_audio, fs)

# 4. Plot the results
time_axis = np.arange(len(raw_audio)) / fs

plt.figure(figsize=(12, 5))
plt.plot(time_axis, raw_audio, color='lightgray', label='Raw Audio')
plt.plot(time_axis, clean_audio, color='blue', alpha=0.8, label='Filtered (25-400 Hz)')
plt.title("Patient a0001: PCG Filtering")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()