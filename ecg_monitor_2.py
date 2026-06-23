import serial
import time
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from scipy.stats import skew, kurtosis
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica
from statistics import mode, mean
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
import fitz  
import os
import io

# CONFIGURAÇÕES 
PORTA_SERIAL = "COM5"     
BAUD_RATE = 9600
DURACAO = 10               
FS = 100                   
MODELO_PATH = "rf_ecg_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Leitura da serial
def ler_serial(porta, baud, duracao):
    print(f"Lendo {duracao}s de dados da serial ({porta})...")
    ser = serial.Serial(porta, baud, timeout=1)
    dados = []

    inicio = time.time()
    while (time.time() - inicio) < duracao:
        if ser.in_waiting > 0:
            try:
                linha = ser.readline().decode().strip()
                if linha:
                    valor = float(linha)
                    dados.append(valor)
            except:
                continue

    ser.close()
    print(f"{len(dados)} amostras coletadas.")
    return np.array(dados)

# Filtragem do sinal
def filtrar_sinal(sinal, fs):
    # 1) Remove drift de linha de base com filtro high-pass mais agressivo (0.67 Hz)
    #    Ordem 4 + filtfilt = 8a ordem efetiva, elimina wandering baseline
    b_hp, a_hp = signal.butter(4, 0.67 / (fs / 2), btype='high')
    sinal_sem_drift = signal.filtfilt(b_hp, a_hp, sinal)

    # 2) Remove ruído de alta frequência com low-pass em 40 Hz
    b_lp, a_lp = signal.butter(4, 40.0 / (fs / 2), btype='low')
    sinal_filtrado = signal.filtfilt(b_lp, a_lp, sinal_sem_drift)

    return sinal_filtrado

# Gera imagem PNG do gráfico ECG e retorna o caminho do arquivo
def gerar_grafico_ecg(ecg_sinal, rpeaks, fs, caminho_img):
    tempo = np.arange(len(ecg_sinal)) / fs

    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    ax.plot(tempo, ecg_sinal, color='#1a73e8', linewidth=0.8, label='ECG filtrado')

    if rpeaks is not None and len(rpeaks) > 0:
        ax.plot(
            tempo[rpeaks],
            ecg_sinal[rpeaks],
            'rv',  # triângulos vermelhos
            markersize=6,
            label='Picos R'
        )

    ax.set_title("ECG — Última Medição", fontsize=11, fontweight='bold')
    ax.set_xlabel("Tempo (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    fig.savefig(caminho_img, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico ECG salvo em: {caminho_img}")

# Classificação do sinal
def extrair_features(signal_data, fs):
    try:
        _, info = nk.ecg_process(signal_data, sampling_rate=fs)
        rpeaks_dict = info.get('ECG_R_Peaks', None)
        if rpeaks_dict is None:
            rpeaks = nk.ecg_findpeaks(signal_data, sampling_rate=fs)['ECG_R_Peaks']
        else:
            rpeaks = np.array(list(rpeaks_dict.values()))
    except Exception:
        rpeaks, _ = nk.ecg_peaks(signal_data, sampling_rate=fs)
        rpeaks = np.where(rpeaks['ECG_R_Peaks'] == 1)[0]

    if len(rpeaks) < 2:
        print("Poucos batimentos detectados.")
        return None

    features = []
    pre_s = int(0.3 * fs)
    post_s = int(0.4 * fs)

    for i in range(1, len(rpeaks)):
        r_idx = rpeaks[i]
        prev_r_idx = rpeaks[i-1]

        rr_interval = (r_idx - prev_r_idx) / fs
        hr = 60.0 / rr_interval if rr_interval > 0 else np.nan
        r_amp = float(signal_data[r_idx])

        start = max(0, r_idx - pre_s)
        end = min(len(signal_data), r_idx + post_s)
        win = signal_data[start:end]

        w_mean = float(np.mean(win))
        w_std = float(np.std(win))
        w_min = float(np.min(win))
        w_max = float(np.max(win))
        w_skew = float(skew(win))
        w_kurt = float(kurtosis(win))
        f, Pxx = signal.welch(win, fs=fs, nperseg=min(256, len(win)))
        spectral_energy = float(np.trapezoid(Pxx, f))

        features.append([
            rr_interval, hr, r_amp, w_mean, w_std, w_min, w_max,
            w_skew, w_kurt, spectral_energy
        ])

    colunas = [
        'rr_interval', 'heart_rate', 'r_amp', 'window_mean', 'window_std',
        'window_min', 'window_max', 'window_skew', 'window_kurt', 'spectral_energy'
    ]
    return pd.DataFrame(features, columns=colunas)

# Execução do código
def main():
    nome_paciente = input("Digite o nome do paciente: ")
    resultados_classes = []
    resultados_bpm = []
    ecg_final = None
    rpeaks_final = None
    bpm_final_medido = 0

    NUM_MEDICOES = 10  # 🔹 número de medições consecutivas

    for i in range(NUM_MEDICOES):
        print(f"\n========== MEDIÇÃO {i+1}/{NUM_MEDICOES} ==========\n")

        # Ler dados crus
        ecg = ler_serial(PORTA_SERIAL, BAUD_RATE, DURACAO)
        if len(ecg) < FS * 2:
            print("Sinal muito curto. Tente novamente.")
            continue

        # Filtrar
        ecg_filtrado = filtrar_sinal(ecg, FS)

        # Detectar picos R
        try:
            _, info = nk.ecg_process(ecg_filtrado, sampling_rate=FS)
            rpeaks_dict = info.get('ECG_R_Peaks', None)
            if rpeaks_dict is None:
                rpeaks = nk.ecg_findpeaks(ecg_filtrado, sampling_rate=FS)['ECG_R_Peaks']
            else:
                rpeaks = np.array(list(rpeaks_dict.values()))
        except Exception:
            rpeaks, _ = nk.ecg_peaks(ecg_filtrado, sampling_rate=FS)
            rpeaks = np.where(rpeaks['ECG_R_Peaks'] == 1)[0]

        # Calcular BPM médio da medição
        if len(rpeaks) > 1:
            rr_intervals = np.diff(rpeaks) / FS
            bpm_medio = 60 / np.mean(rr_intervals)
        else:
            bpm_medio = 0

        # Extrair features
        df_feats = extrair_features(ecg_filtrado, FS)
        if df_feats is None or df_feats.empty:
            print("Falha ao extrair features.")
            continue

        # Carregar modelo e label encoder
        modelo = joblib.load(MODELO_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)

        # Fazer predições e probabilidades
        preds = modelo.predict(df_feats)
        probs = modelo.predict_proba(df_feats)
        labels = le.inverse_transform(preds)

        # Agrupar classes raras como "A"
        labels_simplificados = []
        for lbl in labels:
            if lbl in ['E', 'J', 'F', 'L', 'l']:
                labels_simplificados.append('A')
            elif lbl == 'V':
                labels_simplificados.append('V')
            else:
                labels_simplificados.append('N')

        # Classe final da medição = última (mais recente)
        label_ultimo = labels_simplificados[-1]

        print(f"Resultado da medição {i+1}: Classe {label_ultimo}, BPM médio: {bpm_medio:.1f}")

        # Armazenar resultados
        resultados_classes.append(label_ultimo)
        resultados_bpm.append(bpm_medio)

        # Guardar última medição para o gráfico final
        ecg_final = ecg_filtrado
        rpeaks_final = rpeaks
        bpm_final_medido = bpm_medio

        # Pequena pausa entre medições
        if i < NUM_MEDICOES - 1:
            print("\nAguardando 3 segundos antes da próxima medição...\n")
            time.sleep(3)

    # ====== RESUMO FINAL ======
    if resultados_classes:
        classe_final = mode(resultados_classes)
        bpm_final = mean(resultados_bpm)

        pasta_destino = r"C:\Users\guilh\Documents\RelatoriosECG"
        os.makedirs(pasta_destino, exist_ok=True)

        # --- Gerar gráfico ECG da última medição ---
        caminho_grafico = os.path.join(pasta_destino, "ecg_ultima_medicao.png")
        if ecg_final is not None:
            gerar_grafico_ecg(ecg_final, rpeaks_final, FS, caminho_grafico)

        # --- Preencher o PDF modelo ---
        doc = fitz.open(r"C:\Users\guilh\Downloads\djao.pdf")

        # Página 0 — nome do paciente
        page0 = doc[0]
        page0.insert_text(
            fitz.Point(130, 268),
            nome_paciente,
            fontsize=13.2,
            fontname="helv",
            color=(1, 1, 1)
        )

        # Página 0 — BPM médio geral
        page0.insert_text(
            fitz.Point(32, 90),
            f"BPM médio geral: {bpm_final:.1f}",
            fontsize=13.2,
            fontname="helv",
            color=(0, 0, 0)
        )

        # Página 1 — resultados detalhados
        page1 = doc[1]
        y = 90  # posição vertical inicial
        espacamento = 20  # espaço entre linhas

        linhas_pagina1 = [
            f"Classes detectadas: {resultados_classes}",
            f"BPMs médios: {[round(b, 1) for b in resultados_bpm]}",
            f"Classe final (mais provável): {classe_final}",
        ]

        for linha in linhas_pagina1:
            page1.insert_text(
                fitz.Point(32, y),
                linha,
                fontsize=11,
                fontname="helv",
                color=(0, 0, 0)
            )
            y += espacamento

        # Página 1 — inserir gráfico ECG abaixo do texto
        if ecg_final is not None and os.path.exists(caminho_grafico):
            # Área do gráfico: largura quase total da página, altura proporcional
            rect_grafico = fitz.Rect(30, y + 10, 565, y + 210)
            page1.insert_image(rect_grafico, filename=caminho_grafico)
            print("Gráfico ECG inserido no PDF.")

        # Salvar PDF final
        caminho_saida = os.path.join(pasta_destino, "Relatorio_ECG.pdf")
        doc.save(caminho_saida)
        doc.close()

        print("Arquivo salvo em:", caminho_saida)

    else:
        print("Nenhuma medição válida foi obtida.")

if __name__ == "__main__":
    main()
