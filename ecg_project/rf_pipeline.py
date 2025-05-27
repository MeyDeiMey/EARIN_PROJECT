# === rf_pipeline.py ===
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from models.rf_model import build_rf_model
import scipy.io

CLASSES = ["Normal", "AFib", "MI", "Other"]
data_path = "ecg_project/data/cpsc_dataset"
os.makedirs("images/rf", exist_ok=True)

# Cargar y simular etiquetas
def load_raw_data():
    X, y = [], []
    for file in os.listdir(data_path):
        if not file.endswith(".mat"): continue
        try:
            rec = scipy.io.loadmat(os.path.join(data_path, file))
            signal = rec['ECG'][0, 0][2]
            if signal.shape[0] == 12: signal = signal.T
            if signal.shape[1] != 12 or signal.shape[0] < 100: continue
            if signal.shape[0] > 5000:
                signal = signal[:5000, :]
            else:
                pad = np.zeros((5000 - signal.shape[0], 12))
                signal = np.vstack([signal, pad])
            X.append(signal)
            y.append(random.choice(CLASSES))
        except: continue
    return np.array(X), np.array(y)

# Extraer features básicos
def extract_features(X):
    feats = []
    for s in X:
        f = []
        for i in range(12):
            lead = s[:, i]
            f.extend([np.mean(lead), np.std(lead), np.max(lead), np.min(lead), np.sum(lead**2)])
        feats.append(f)
    return np.array(feats)

print("📦 RF: Cargando y procesando datos...")
X_raw, y = load_raw_data()
X = extract_features(X_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = build_rf_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Random Forest Confusion Matrix")
plt.savefig("images/rf/confusion_matrix.png")
print("✅ Resultados guardados en images/rf/confusion_matrix.png")
