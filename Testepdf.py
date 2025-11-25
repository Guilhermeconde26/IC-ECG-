import numpy as np
import joblib
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
import fitz  
import os
import matplotlib.pyplot as plt
import io

modelo = joblib.load("rf_ecg_model.pkl")
le = joblib.load("label_encoder.pkl")
n = modelo.n_features_in_

feat = [
    "RR_interval", "Heart_rate", "r_amp", "Window_mean", "Window_std",
    "Window_min", "Window_max", "Window_skew", "Window_Kurt", "Spectral_energy"
]

novas_features = np.array([
    [0.68, 88.24, 153.99, -1.49, 28.51, -26.28, 153.98, 3.80, 16.02, 1801.79]
])

preds = modelo.predict(novas_features)
probs = modelo.predict_proba(novas_features)
classes_previstas = le.inverse_transform(preds)


doc = fitz.open("PLT.pdf")
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

eixo = 90
eixo2 = 570

for i, (classe, feat_row, prob_row) in enumerate(zip(classes_previstas, novas_features, probs)):
    for nome, valor in zip(feat, feat_row):
        h = (f"  {nome:<15} = {valor}")
        page = doc[1]
        ponto = fitz.Point(30, eixo)  
        texto = h
        page.insert_text(
            ponto,
            texto,
            fontsize=10.6,
            fontname="helv",  # Fonte 
            color=(0, 0, 0)   # Cor em RGB 
        )
        eixo += 20

    classep= (f"\n Classe prevista: {classe}")
    page = doc[1]
    ponto = fitz.Point(32, 510)  
    texto = classep
    page.insert_text(
        ponto,
        texto,
        fontsize=10.6,
        fontname="helv",  # Fonte 
        color=(0, 0, 0)   # Cor em RGB 
    )

    idx_sorted = np.argsort(prob_row)[::-1]
    prob=("\n Probabilidades (top 3):")
    classep= (f"\n Classe prevista: {classe}")
    page = doc[1]
    ponto = fitz.Point(32, 530)  
    texto = prob
    page.insert_text(
        ponto,
        texto,
        fontsize=10.6,
        fontname="helv",  # Fonte 
        color=(0, 0, 0)   # Cor em RGB 
    )

    for idx in idx_sorted[:3]:
        yu = (f" {le.classes_[idx]}: {prob_row[idx]:.3f}")
        page = doc[1]
        ponto = fitz.Point(32, eixo2)  
        texto = yu
        page.insert_text(
            ponto,
            texto,
            fontsize=10.6,
            fontname="helv",  # Fonte 
            color=(0, 0, 0)   # Cor em RGB 
         )
        eixo2 += 20

doc.save("Teste23.pdf")
doc.close()
