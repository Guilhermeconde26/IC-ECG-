import serial
import time
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from scipy.stats import skew, kurtosis
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statistics import mode, mean
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import os

# ── CONFIGURAÇÕES ──────────────────────────────────────────────
PORTA_SERIAL      = "COM5"
BAUD_RATE         = 9600
DURACAO           = 10
FS                = 100
MODELO_PATH       = "rf_ecg_model.pkl"
LABEL_ENCODER_PATH= "label_encoder.pkl"
PASTA_DESTINO     = r"C:\Users\guilh\Documents\RelatoriosECG"
NUM_MEDICOES      = 10

# ── CORES DO LAYOUT (fiel ao modelo) ───────────────────────────
COR_FUNDO_TITULO  = colors.HexColor("#8B0000")   # vermelho escuro
COR_FUNDO_CINZA   = colors.HexColor("#4A4A4A")   # cinza escuro
COR_BRANCO        = colors.white
COR_PRETO         = colors.black
COR_LINHA         = colors.HexColor("#8B0000")

# ── LEITURA SERIAL ─────────────────────────────────────────────
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
                    dados.append(float(linha))
            except:
                continue
    ser.close()
    print(f"{len(dados)} amostras coletadas.")
    return np.array(dados)

# ── FILTRO ─────────────────────────────────────────────────────
def filtrar_sinal(sinal, fs):
    b_hp, a_hp = signal.butter(4, 0.67 / (fs / 2), btype='high')
    sinal_sem_drift = signal.filtfilt(b_hp, a_hp, sinal)
    b_lp, a_lp = signal.butter(4, 40.0 / (fs / 2), btype='low')
    return signal.filtfilt(b_lp, a_lp, sinal_sem_drift)

# ── GRÁFICO ECG ────────────────────────────────────────────────
def gerar_grafico_ecg(ecg_sinal, rpeaks, fs, caminho_img, titulo="ECG — Última Medição"):
    tempo = np.arange(len(ecg_sinal)) / fs
    fig, ax = plt.subplots(figsize=(9, 2.5), dpi=150)
    ax.plot(tempo, ecg_sinal, color='#1a73e8', linewidth=0.8, label='ECG filtrado')
    if rpeaks is not None and len(rpeaks) > 0:
        ax.plot(tempo[rpeaks], ecg_sinal[rpeaks], 'rv', markersize=5, label='Picos R')
    ax.set_title(titulo, fontsize=10, fontweight='bold')
    ax.set_xlabel("Tempo (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho_img, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ── EXTRAÇÃO DE FEATURES ───────────────────────────────────────
def extrair_features(signal_data, fs):
    try:
        _, info = nk.ecg_process(signal_data, sampling_rate=fs)
        rpeaks_dict = info.get('ECG_R_Peaks', None)
        rpeaks = np.array(list(rpeaks_dict.values())) if rpeaks_dict else \
                 nk.ecg_findpeaks(signal_data, sampling_rate=fs)['ECG_R_Peaks']
    except Exception:
        rpeaks_df, _ = nk.ecg_peaks(signal_data, sampling_rate=fs)
        rpeaks = np.where(rpeaks_df['ECG_R_Peaks'] == 1)[0]

    if len(rpeaks) < 2:
        print("Poucos batimentos detectados.")
        return None

    features = []
    pre_s  = int(0.3 * fs)
    post_s = int(0.4 * fs)

    for i in range(1, len(rpeaks)):
        r_idx      = rpeaks[i]
        prev_r_idx = rpeaks[i - 1]
        rr_interval = (r_idx - prev_r_idx) / fs
        hr          = 60.0 / rr_interval if rr_interval > 0 else np.nan
        r_amp       = float(signal_data[r_idx])
        start = max(0, r_idx - pre_s)
        end   = min(len(signal_data), r_idx + post_s)
        win   = signal_data[start:end]
        f, Pxx = signal.welch(win, fs=fs, nperseg=min(256, len(win)))
        features.append([
            rr_interval, hr, r_amp,
            float(np.mean(win)), float(np.std(win)),
            float(np.min(win)),  float(np.max(win)),
            float(skew(win)),    float(kurtosis(win)),
            float(np.trapezoid(Pxx, f))
        ])

    colunas = ['rr_interval','heart_rate','r_amp','window_mean','window_std',
               'window_min','window_max','window_skew','window_kurt','spectral_energy']
    return pd.DataFrame(features, columns=colunas)

# ── GERAÇÃO DO RELATÓRIO PDF ────────────────────────────────────
def gerar_relatorio_pdf(nome_paciente, resultados_classes, resultados_bpm,
                        classe_final, bpm_final, fs,
                        graficos,           # lista de caminhos PNG (últimas 3)
                        caminho_saida):

    W, H = A4  # 595 x 842 pt
    c = canvas.Canvas(caminho_saida, pagesize=A4)

    # ── CABEÇALHO: faixa vermelha com título ──
    c.setFillColor(COR_FUNDO_TITULO)
    c.rect(0, H - 70, W, 70, fill=1, stroke=0)
    c.setFillColor(COR_BRANCO)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(W / 2, H - 45, "ECG VIA ESP32-S3")

    # ── SUBTÍTULO ──
    c.setFillColor(COR_PRETO)
    c.setFont("Helvetica", 9)
    c.drawCentredString(W / 2, H - 82, "ALUNO: GUILHERME,  PROF: ADRIELLE")

    # ── FAIXA CINZA: NOME e RESPONSÁVEL ──
    c.setFillColor(COR_FUNDO_CINZA)
    c.rect(0, H - 115, W, 28, fill=1, stroke=0)
    c.setFillColor(COR_BRANCO)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20, H - 107, f"NOME:  {nome_paciente.upper()}")
    c.drawString(320, H - 107, "RESPONSAVEL:  MEDICO")

    # ── TÍTULO SEÇÃO GRÁFICOS ──
    c.setFillColor(COR_PRETO)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(W / 2, H - 140, "GRAFICO DA MEDIDA")
    # linha decorativa
    c.setStrokeColor(COR_LINHA)
    c.setLineWidth(1.5)
    c.line(30, H - 145, W - 30, H - 145)

    # ── 3 GRÁFICOS DAS ÚLTIMAS MEDIÇÕES ──
    altura_grafico = 130   # pt
    margem_lateral = 25
    largura_grafico = W - 2 * margem_lateral
    y_inicio = H - 155

    for idx, caminho_png in enumerate(graficos):
        y_topo = y_inicio - idx * (altura_grafico + 8)
        y_base = y_topo - altura_grafico
        if os.path.exists(caminho_png):
            c.drawImage(caminho_png,
                        margem_lateral, y_base,
                        width=largura_grafico, height=altura_grafico,
                        preserveAspectRatio=True)

    # ── SEÇÃO DE INFORMAÇÕES ──
    y_info = y_inicio - 3 * (altura_grafico + 8) - 20

    c.setStrokeColor(COR_LINHA)
    c.setLineWidth(1)
    c.line(30, y_info + 5, W - 30, y_info + 5)

    c.setFillColor(COR_PRETO)
    c.setFont("Helvetica-Bold", 11)
    espacamento = 22

    velocidade_amostragem = fs        # amostras/s = Hz
    periodo_amostragem    = 1.0 / fs  # segundos por amostra

    linhas_info = [
        f"INFORMACAO DE FREQUENCIA DE AMOSTRAGEM:  {fs} Hz",
        f"INFORMACAO DE VELOCIDADE DE AMOSTRAGEM:  {periodo_amostragem*1000:.1f} ms/amostra",
        f"CLASSE PREVISTA:  {classe_final}",
        f"MEDIDA DE BATIMENTOS POR MINUTO:  {bpm_final:.1f} BPM",
    ]

    y_texto = y_info - 10
    for linha in linhas_info:
        c.drawString(30, y_texto, linha)
        y_texto -= espacamento

    # ── LOGO UFOP (canto inferior direito) ──
    c.setFillColor(COR_FUNDO_TITULO)
    c.rect(W - 70, 0, 70, 55, fill=1, stroke=0)
    c.setFillColor(COR_BRANCO)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(W - 35, 20, "UFOP")

    c.save()
    print(f"Relatório salvo em: {caminho_saida}")

# ── MAIN ───────────────────────────────────────────────────────
def main():
    nome_paciente = input("Digite o nome do paciente: ")
    resultados_classes = []
    resultados_bpm     = []

    # Guardar as últimas 3 medições (sinal + rpeaks)
    ultimas_medicoes = []   # lista de dicts {ecg, rpeaks}

    modelo = joblib.load(MODELO_PATH)
    le     = joblib.load(LABEL_ENCODER_PATH)

    for i in range(NUM_MEDICOES):
        print(f"\n========== MEDIÇÃO {i+1}/{NUM_MEDICOES} ==========\n")

        ecg = ler_serial(PORTA_SERIAL, BAUD_RATE, DURACAO)
        if len(ecg) < FS * 2:
            print("Sinal muito curto. Pulando.")
            continue

        ecg_filtrado = filtrar_sinal(ecg, FS)

        # Detectar picos R
        try:
            _, info = nk.ecg_process(ecg_filtrado, sampling_rate=FS)
            rpeaks_dict = info.get('ECG_R_Peaks', None)
            rpeaks = np.array(list(rpeaks_dict.values())) if rpeaks_dict else \
                     nk.ecg_findpeaks(ecg_filtrado, sampling_rate=FS)['ECG_R_Peaks']
        except Exception:
            rp, _ = nk.ecg_peaks(ecg_filtrado, sampling_rate=FS)
            rpeaks = np.where(rp['ECG_R_Peaks'] == 1)[0]

        # BPM
        bpm_medio = 60 / np.mean(np.diff(rpeaks) / FS) if len(rpeaks) > 1 else 0

        # Features + classificação
        df_feats = extrair_features(ecg_filtrado, FS)
        if df_feats is None or df_feats.empty:
            print("Falha ao extrair features.")
            continue

        preds  = modelo.predict(df_feats)
        labels = le.inverse_transform(preds)

        labels_simplificados = []
        for lbl in labels:
            if lbl in ['E', 'J', 'F', 'L', 'l']:
                labels_simplificados.append('A')
            elif lbl == 'V':
                labels_simplificados.append('V')
            else:
                labels_simplificados.append('N')

        label_medicao = labels_simplificados[-1]
        print(f"Medição {i+1}: Classe={label_medicao}, BPM={bpm_medio:.1f}")

        resultados_classes.append(label_medicao)
        resultados_bpm.append(bpm_medio)

        # Guardar para gráfico (mantém só as últimas 3)
        ultimas_medicoes.append({'ecg': ecg_filtrado, 'rpeaks': rpeaks})
        if len(ultimas_medicoes) > 3:
            ultimas_medicoes.pop(0)

        if i < NUM_MEDICOES - 1:
            print("\nAguardando 3 segundos...\n")
            time.sleep(3)

    # ── RESUMO ──
    if not resultados_classes:
        print("Nenhuma medição válida.")
        return

    classe_final = mode(resultados_classes)
    bpm_final    = mean(resultados_bpm)

    os.makedirs(PASTA_DESTINO, exist_ok=True)

    # Gerar PNGs das últimas 3 medições
    caminhos_graficos = []
    for idx, med in enumerate(ultimas_medicoes):
        n = len(ultimas_medicoes)
        num_medicao = NUM_MEDICOES - (n - 1 - idx)
        caminho_png = os.path.join(PASTA_DESTINO, f"ecg_medicao_{num_medicao}.png")
        gerar_grafico_ecg(
            med['ecg'], med['rpeaks'], FS,
            caminho_png,
            titulo=f"ECG — Medição {num_medicao}"
        )
        caminhos_graficos.append(caminho_png)

    # Gerar PDF
    caminho_pdf = os.path.join(PASTA_DESTINO, "Relatorio_ECG.pdf")
    gerar_relatorio_pdf(
        nome_paciente   = nome_paciente,
        resultados_classes = resultados_classes,
        resultados_bpm  = resultados_bpm,
        classe_final    = classe_final,
        bpm_final       = bpm_final,
        fs              = FS,
        graficos        = caminhos_graficos,
        caminho_saida   = caminho_pdf
    )

if __name__ == "__main__":
    main()
