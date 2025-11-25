import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ------------- ajuste aqui o caminho do CSV se necessário -------------
CSV_PATH = r"C:\Users\guilh\Downloads\ecg_features.csv"
# ----------------------------------------------------------------------

# Carregar CSV
df = pd.read_csv(CSV_PATH, low_memory=False)
print("Dimensão do dataframe:", df.shape)

# 1) Converter label para string e tratar NaN
df["label"] = df["label"].astype(str)
df.loc[df["label"].str.lower() == "nan", "label"] = np.nan

# Remover linhas sem label
df = df.dropna(subset=["label"])
print("Após remover linhas sem label:", df.shape)

# 2) Remover classes raras (<50 amostras)
counts = df['label'].value_counts()
keep_labels = counts[counts >= 50].index
df = df[df['label'].isin(keep_labels)]
print("Após remover classes raras (<50):", df.shape)
print("Classes mantidas:", df['label'].unique())

# 3) Selecionar features numéricas
drop_cols = ["record", "r_sample", "fs"]
X = df.select_dtypes(include=[np.number]).copy()
for c in drop_cols:
    if c in X.columns:
        X = X.drop(columns=[c])

# 4) Preencher NaNs em X com mediana
X = X.fillna(X.median())

# 5) Preparar y e codificar labels
y = df["label"].astype(str).copy()
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes codificadas:")
for cls, code in zip(le.classes_, range(len(le.classes_))):
    print(code, cls)

# 6) Dividir treino/teste com stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 7) Treinar Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 8) Avaliar
y_pred = model.predict(X_test)
print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 9) Salvar modelo e encoder
joblib.dump(model, "rf_ecg_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\nModelo salvo em 'rf_ecg_model.pkl' e encoder em 'label_encoder.pkl'.")
