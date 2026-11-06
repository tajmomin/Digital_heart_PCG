import os
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks

# --- 1. Helper Functions ---
def apply_bandpass_filter(data, fs, lowcut=25.0, highcut=400.0, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, data)

def calculate_shannon_envelope(audio_signal, fs, smooth_window_ms=20):
    # Normalize to [-1, 1]
    norm_audio = audio_signal / np.max(np.abs(audio_signal))
    squared_audio = norm_audio ** 2
    
    # Shannon Energy
    epsilon = 1e-10 
    energy = - (squared_audio) * np.log(squared_audio + epsilon)
    
    # Smooth the envelope
    window_length = int((smooth_window_ms / 1000.0) * fs)
    window = np.ones(window_length) / window_length
    smoothed_envelope = np.convolve(energy, window, mode='same')
    
    return smoothed_envelope / np.max(smoothed_envelope) # Normalize to 0-1

# --- 2. Robust Path Setup ---
# Dynamically points to the Data folder one level above Src
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.abspath(os.path.join(script_dir, "..", "Data", "training-a"))
wav_files = glob.glob(os.path.join(data_folder, "*.wav"))

print(f"Found {len(wav_files)} audio files. Processing the first 5...")

# --- 3. Batch Processing Loop ---
extracted_data = []

for file_path in wav_files[:5]:  # Remove [:5] later to process the whole folder
    filename = os.path.basename(file_path)
    fs, raw_audio = wavfile.read(file_path)
    
    # Process signal
    clean_audio = apply_bandpass_filter(raw_audio, fs)
    envelope = calculate_shannon_envelope(clean_audio, fs)
    
    # Find peaks (S1 and S2 heart sounds)
    # distance=fs*0.2 ensures peaks are at least 200ms apart (physiological minimum)
    # height=0.2 filters out low-level background noise
    peaks, _ = find_peaks(envelope, distance=fs*0.2, height=0.2)
    
    # Calculate timings (in seconds, then convert to milliseconds)
    peak_times_sec = peaks / fs
    
    if len(peak_times_sec) >= 2:
        # For this FYP MVP, we estimate LVET as the median time between consecutive peaks 
        # (A more advanced HSMM model would strictly classify S1 vs S2)
        intervals = np.diff(peak_times_sec)
        
        # S1->S2 is shorter (systole), S2->S1 is longer (diastole)
        # We sort intervals and take the median of the shorter half as LVET
        systolic_intervals = intervals[intervals < np.mean(intervals)]
        
        lvet_ms = np.median(systolic_intervals) * 1000 if len(systolic_intervals) > 0 else 0
        heart_rate_bpm = 60.0 / np.median(intervals[intervals > np.mean(intervals)]) if len(intervals) > 2 else 0
        
        extracted_data.append({
            "Patient_ID": filename.replace(".wav", ""),
            "Heart_Rate_BPM": round(heart_rate_bpm, 1),
            "LVET_ms": round(lvet_ms, 1),
            "Num_Peaks_Detected": len(peaks)
        })

# --- 4. Export to CSV ---
df = pd.DataFrame(extracted_data)
output_csv = os.path.join(script_dir, "..", "Data", "patient_timings.csv")
df.to_csv(output_csv, index=False)

print(f"\nProcessing complete! Data saved to:\n{os.path.abspath(output_csv)}")
print(df)