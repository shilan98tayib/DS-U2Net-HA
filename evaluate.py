"""
Speech Enhancement Evaluation Script
This script evaluates the average values of evaluation metrics of final enhanced speech using the following:
- PESQ
- STOI
- CSIG
- CBAK
- COVL

It also exports the evaluation results to an Excel file and
randomly visualizes waveform and spectrogram examples.
"""


import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from pesq import pesq
from pystoi import stoi
import pandas as pd
import speech_recognition as sr
import config as cfg
# ✅ Composite measures (CSIG, CBAK, COVL)
# pip install https://github.com/schmiph2/pysepm/archive/master.zip
from pysepm.qualityMeasures import composite


# =========================
# Paths
# =========================
noisy_dir = "PATH_TO_NOISY_DATASET"
clean_dir = "PATH_TO_CLEAN_DATASET"
enhanced_dir = "PATH_TO_ENHANCED_RESULTS"
excel_out_path = os.path.join(enhanced_dir,'FINAL_RESULT.xlsx')

ENABLE_DRAWING = True
RANDOM_SEED = cfg.SEED

results = []


# ==========================================================
# UTILITIES
# ==========================================================

def load_audio(file_path, sr=cfg.FS):
    signal, _ = librosa.load(file_path, sr=sr, mono=True)
    return signal


def pad_signals(ref, target):
    if len(ref) > len(target):
        target = np.pad(target, (0, len(ref) - len(target)))
    elif len(ref) < len(target):
        ref = np.pad(ref, (0, len(target) - len(ref)))
    return ref, target


# ==========================================================
# PLOTTING FUNCTIONS
# ==========================================================

def plot_spectrograms(noisy, clean, enhanced, filename, sr=cfg.FS):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    titles = ['Noisy Speech', 'Clean Speech', 'Enhanced Speech']
    signals = [noisy, clean, enhanced]

    for i, (sig, title) in enumerate(zip(signals, titles)):
        f, t, Sxx = spectrogram(sig, fs=sr)
        axes[i].imshow(10 * np.log10(Sxx + 1e-10),
                       aspect='auto', origin='lower', cmap='inferno')
        axes[i].set_title(f"{title} - {filename}")
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


def plot_waveforms(noisy, clean, enhanced, filename, sr=cfg.FS):
    t_noisy = np.arange(len(noisy)) / sr
    t_clean = np.arange(len(clean)) / sr
    t_enh   = np.arange(len(enhanced)) / sr

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t_noisy, noisy)
    axes[0].set_title(f"Noisy - {filename}")

    axes[1].plot(t_clean, clean)
    axes[1].set_title("Clean")

    axes[2].plot(t_enh, enhanced)
    axes[2].set_title("Enhanced")

    axes[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ==========================================================
# MAIN PROCESSING
# ==========================================================

file_names = [f for f in os.listdir(noisy_dir) if f.endswith('.wav')]

for i, fname in enumerate(file_names, 1):
    print(f"Processing {i}/{len(file_names)} : {fname}")

    noisy_path = os.path.join(noisy_dir, fname)
    clean_path = os.path.join(clean_dir, fname)
    enhanced_path = os.path.join(enhanced_dir, 'enhanced_' + fname)

    if not (os.path.exists(clean_path) and os.path.exists(enhanced_path)):
        continue

    noisy = load_audio(noisy_path)
    clean = load_audio(clean_path)
    enhanced = load_audio(enhanced_path)

    clean, noisy = pad_signals(clean, noisy)
    clean, enhanced = pad_signals(clean, enhanced)

    try:
        pesq_score = pesq(cfg.FS, clean, enhanced, 'wb')
    except:
        pesq_score = np.nan

    try:
        stoi_score = stoi(clean, enhanced, cfg.FS)
    except:
        stoi_score = np.nan

    try:
        csig, cbak, covl = composite(clean, enhanced, cfg.FS)
    except:
        csig, cbak, covl = np.nan, np.nan, np.nan


    results.append({
        "file": fname,
        "PESQ": pesq_score,
        "STOI": stoi_score,
        "CSIG": csig,
        "CBAK": cbak,
        "COVL": covl
    })


# ==========================================================
# SAVE RESULTS
# ==========================================================

df = pd.DataFrame(results)

means = df.mean(numeric_only=True)

print("\n=== Averages ===")
for metric in means.index:
    print(f"{metric}: {means[metric]:.4f}")

df = pd.concat([df, pd.DataFrame([{
    "file": "MEAN",
    **means.to_dict()
}])], ignore_index=True)

os.makedirs(os.path.dirname(excel_out_path), exist_ok=True)
df.to_excel(excel_out_path, index=False)
print(f"\nResults saved to: {excel_out_path}")


# ==========================================================
# RANDOM SAMPLE DRAWING
# ==========================================================

valid_samples = df[(df["file"] != "MEAN") & (df["PESQ"].notna())]

if len(valid_samples) > 0:

    random_row = valid_samples.sample(n=1, random_state=RANDOM_SEED).iloc[0]
    random_file = random_row["file"]

    print("\n=== Random Sample Selected ===")
    print(random_row)

    if ENABLE_DRAWING:
        noisy_r = load_audio(os.path.join(noisy_dir, random_file))
        clean_r = load_audio(os.path.join(clean_dir, random_file))
        enhanced_r = load_audio(os.path.join(enhanced_dir, 'enhanced_' + random_file))

        clean_r, noisy_r = pad_signals(clean_r, noisy_r)
        clean_r, enhanced_r = pad_signals(clean_r, enhanced_r)

        plot_spectrograms(noisy_r, clean_r, enhanced_r, random_file)
        plot_waveforms(noisy_r, clean_r, enhanced_r, random_file)

else:
    print("No valid samples found for random drawing.")




