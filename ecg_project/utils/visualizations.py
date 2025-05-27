import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion(y_true, y_pred, labels):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
