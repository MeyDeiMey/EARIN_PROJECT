# === extract_features.py ===
import numpy as np

def extract_features_from_dataset(X_raw):
    features = []
    for signal in X_raw:
        signal_features = []
        for i in range(12):
            lead = signal[:, i]
            signal_features.extend([
                np.mean(lead),
                np.std(lead),
                np.max(lead),
                np.min(lead),
                np.sum(lead ** 2)
            ])
        features.append(signal_features)
    return np.array(features)
