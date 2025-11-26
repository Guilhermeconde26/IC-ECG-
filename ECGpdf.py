import serial
import time
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from scipy.stats import skew, kurtosis
import joblib
import matplotlib.pyplot as plt
from statistics import mode, mean
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
import fitz  
import os
import io

# CONFIGURAÇÕES 
PORTA_SERIAL = "COM5"      # ajustar sempre que conectar o ECG 
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
    b, a = signal.butter(4, [0.5/(fs/2), 40/(fs/2)], btype='band')
    return signal.filtfilt(b, a, sinal)

# Classificaçao do sinal
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

#Execuçao do codigo
def main():
    resultados_classes = []
    resultados_bpm = []
    ecg_final = None
    rpeaks_final = None
    bpm_final_medido = 0

    NUM_MEDICOES = 5  # 🔹 número de medições consecutivas

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
        print("\nAguardando 3 segundos antes da próxima medição...\n")
        time.sleep(3)

    # ====== RESUMO FINAL ======
    if resultados_classes:
        classe_final = mode(resultados_classes)
        bpm_final = mean(resultados_bpm)

        doc = fitz.open(r"C:\Users\guilh\Downloads\djao.pdf")
        nome_paciente = input("Digite o nome do paciente: ")
        page = doc[0]
        ponto = fitz.Point(130, 268)  
        texto = nome_paciente


        page.insert_text(
            ponto,
            texto,
            fontsize=13.2,
            fontname="helv",  # Fonte 
            color=(1, 1, 1)   # Cor em RGB 
        )
        classe=(f"Classes detectadas: {resultados_classes}")
        page = doc[1]
        ponto = fitz.Point(32, 90)  
        texto = classe


        page.insert_text(
            ponto,
            texto,
            fontsize=13.2,
            fontname="helv",  # Fonte 
            color=(0, 0, 0)   # Cor em RGB 
        )
        batimento=(f"BPMs médios: {[round(b,1) for b in resultados_bpm]}")
        page = doc[1]
        ponto = fitz.Point(32, 90)  
        texto = batimento


        page.insert_text(
            ponto,
            texto,
            fontsize=13.2,
            fontname="helv",  # Fonte 
            color=(0, 0,0)   # Cor em RGB 
        )
        classeprob=(f"Classe final (mais provável): {classe_final}")
        page = doc[1]
        ponto = fitz.Point(32, 90)  
        texto = classeprob


        page.insert_text(
            ponto,
            texto,
            fontsize=13.2,
            fontname="helv",  # Fonte 
            color=(0, 0, 0)   # Cor em RGB 
        )
        batigera= (f"BPM médio geral: {bpm_final:.1f}")
        page = doc[0]
        ponto = fitz.Point(32, 90)  
        texto = batigera


        page.insert_text(
            ponto,
            texto,
            fontsize=13.2,
            fontname="helv",  # Fonte 
            color=(0, 0, 0)   # Cor em RGB 
        )
       

        # 🔹 Plotar gráfico apenas da última medição
        if ecg_final is not None:
            tempo = np.arange(len(ecg_final)) / FS
            plt.figure(figsize=(10, 4))
            plt.plot(tempo, ecg_final, label="ECG Filtrado", linewidth=1)
            plt.scatter(rpeaks_final / FS, ecg_final[rpeaks_final], color='red', marker='o', label='Picos R')
            plt.title(f"Sinal ECG (Última Medição) — BPM: {bpm_final_medido:.1f}")
            plt.xlabel("Tempo (s)")
            plt.ylabel("Amplitude (mV)")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            plt.close() 
            rect_grafico = fitz.Rect(100, 200, 500, 400) 

            page.insert_image(
                rect_grafico,
                stream=img_buffer 
            )

            print("Gráfico Matplotlib gerado em memória.")
            pasta_destino = r"C:\Users\guilh\Documents\RelatoriosECG"

            os.makedirs(pasta_destino, exist_ok=True)

            caminho_saida = os.path.join(pasta_destino, "Relatorio_ECG.pdf")

            doc.save(caminho_saida)
            
            doc.close()

            print("Arquivo salvo em:", caminho_saida)


    else:
        print("Nenhuma medição válida foi obtida.")

if __name__ == "__main__":
    main()
