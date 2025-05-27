# === main.py ===
import subprocess
import time

pipelines = [
    ("Random Forest", "ecg_project/rf_pipeline.py"),
    ("CNN (Keras)", "ecg_project/cnn_pipeline.py"),
    ("LSTM (Keras)", "ecg_project/lstm_pipeline.py")
]

for name, script in pipelines:
    print(f"\n🚀 Ejecutando {name}...")
    start = time.time()
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando {script}:\n{e}")
    end = time.time()
    print(f"✅ {name} terminado en {round(end - start, 2)} segundos")
