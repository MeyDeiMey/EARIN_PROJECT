# === cnn_pipeline.py === (Keras)
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import spectrogram
import scipy.io
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.utils import to_categorical

CLASSES = ["Normal", "AFib", "MI", "Other"]
data_path = "ecg_project/data/cpsc_dataset"
os.makedirs("images/cnn", exist_ok=True)

# Cargar señales y simular etiquetas
def load_data():
    X, y = [], []
    for f in os.listdir(data_path):
        if not f.endswith(".mat"): continue
        try:
            rec = scipy.io.loadmat(os.path.join(data_path, f))
            s = rec['ECG'][0, 0][2]
            if s.shape[0] == 12: s = s.T
            if s.shape[1] != 12 or s.shape[0] < 100: continue
            if s.shape[0] > 5000:
                s = s[:5000, :]
            else:
                pad = np.zeros((5000 - s.shape[0], 12))
                s = np.vstack([s, pad])
            # Extraemos una señal media de todas las derivaciones
            X.append(np.mean(s, axis=1))
            y.append(random.choice(CLASSES))
        except: continue
    return np.array(X), np.array(y)

# Generar espectrogramas (convertir a imágenes)
def get_spectrograms(X):
    specs = []
    for sig in X:
        f, t, Sxx = spectrogram(sig, fs=500)
        padded = np.zeros((128, 128))
        cropped = Sxx[:128, :128]
        padded[:cropped.shape[0], :cropped.shape[1]] = cropped
        specs.append(padded)
    return np.expand_dims(np.array(specs), axis=-1)

print("🎛 CNN (Keras): Preparando datos...")
X_raw, y = load_data()
X_spec = get_spectrograms(X_raw)
y_idx = np.array([CLASSES.index(label) for label in y])
y_cat = to_categorical(y_idx, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X_spec, y_cat, test_size=0.2, random_state=42)

# Modelo CNN corregido
model = Sequential([
    Input(shape=(128, 128, 1)),
    Conv2D(16, kernel_size=3, activation='relu', padding='same'),
    MaxPooling2D(2),
    Conv2D(32, kernel_size=3, activation='relu', padding='same'),
    MaxPooling2D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("📟 Entrenando CNN...")
model.fit(X_train, y_train, epochs=5, batch_size=8, verbose=1)

# Evaluación
preds = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(preds, axis=1)
print(classification_report(y_true, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("CNN Confusion Matrix (Keras)")
plt.tight_layout()
plt.savefig("images/cnn/confusion_matrix.png")
print("✅ Resultados guardados en images/cnn/")