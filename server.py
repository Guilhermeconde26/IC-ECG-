from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from scipy.stats import skew, kurtosis
import joblib

app = FastAPI()

# ===== CONFIG =====
FS = 100   # ajuste se seu ESP estiver em outra taxa
MODELO_PATH = "rf_ecg_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# ===== LOAD MODEL (UMA VEZ) =====
modelo = joblib.load(MODELO_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

class ECGInput(BaseModel):
    signal: list[float]

# ================================
# FILTRO
# ================================
def filtrar_sinal(sinal, fs):
    b, a = signal.butter(3, [0.5/(fs/2), 40/(fs/2)], btype='band')
    return signal.filtfilt(b, a, sinal)

# ================================
# EXTRAÇÃO DE FEATURES
# ================================
def extrair_features(signal_data, rpeaks, fs):

    if len(rpeaks) < 2:
        return None

    features = []
    pre_s = int(0.3 * fs)
    post_s = int(0.4 * fs)

    for i in range(1, len(rpeaks)):
        r_idx = int(rpeaks[i])
        prev_r_idx = int(rpeaks[i-1])

        rr_interval = (r_idx - prev_r_idx) / fs
        if rr_interval <= 0:
            continue

        hr = 60.0 / rr_interval
        r_amp = float(signal_data[r_idx])

        start = max(0, r_idx - pre_s)
        end = min(len(signal_data), r_idx + post_s)
        win = signal_data[start:end]

        if len(win) < 10:
            continue

        f, Pxx = signal.welch(win, fs=fs, nperseg=min(256, len(win)))
        spectral_energy = float(np.trapezoid(Pxx, f))

        features.append([
            rr_interval,
            hr,
            r_amp,
            float(np.mean(win)),
            float(np.std(win)),
            float(np.min(win)),
            float(np.max(win)),
            float(skew(win)),
            float(kurtosis(win)),
            spectral_energy
        ])

    if len(features) == 0:
        return None

    colunas = [
        'rr_interval','heart_rate','r_amp','window_mean','window_std',
        'window_min','window_max','window_skew','window_kurt','spectral_energy'
    ]

    return pd.DataFrame(features, columns=colunas)

# ================================
# ENDPOINT
# ================================
@app.post("/classify")
async def classify(data: ECGInput):

    ecg = np.array(data.signal, dtype=np.float32)

    print("Tamanho do sinal:", len(ecg))
    print("Min:", np.min(ecg))
    print("Max:", np.max(ecg))
    print("Mean:", np.mean(ecg))
    print("Std:", np.std(ecg))

    if len(ecg) < FS * 2:
        return {"erro": "Sinal muito curto"}

    try:
        # ===== FILTRAR =====
        ecg_filtrado = filtrar_sinal(ecg, FS)

        # ===== REMOVER OFFSET =====
        ecg_filtrado = ecg_filtrado - np.mean(ecg_filtrado)

        # ===== NORMALIZAR =====
        std = np.std(ecg_filtrado)
        if std < 1e-6:
            return {"erro": "Sinal muito fraco"}
        ecg_filtrado = ecg_filtrado / std

        # ===== DETECTAR R-PEAKS =====
        signals, info = nk.ecg_peaks(
            ecg_filtrado,
            sampling_rate=FS,
            method="pantompkins1985"
        )

        rpeaks = np.array(info["ECG_R_Peaks"])

        print("R-peaks detectados:", len(rpeaks))

        if len(rpeaks) < 2:
            return {"erro": "Poucos R-peaks detectados"}

    except Exception as e:
        print("Erro no neurokit:", e)
        return {"erro": "Falha na extração"}

    # ===== FEATURES =====
    df_feats = extrair_features(ecg_filtrado, rpeaks, FS)

    if df_feats is None or df_feats.empty:
        return {"erro": "Falha na extração de features"}

    # ===== CLASSIFICAÇÃO =====
    try:
        preds = modelo.predict(df_feats)
        labels = le.inverse_transform(preds)
        classe_final = labels[-1]
    except Exception as e:
        print("Erro no modelo:", e)
        return {"erro": "Falha na classificação"}

    # ===== BPM =====
    rr_intervals = np.diff(rpeaks) / FS
    bpm = 60 / np.mean(rr_intervals)

    return {
        "classe": classe_final,
        "bpm": float(round(bpm, 1))
    }