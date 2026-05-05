import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the confusion matrix from running the baseline.py file
cm = np.array([
    [38, 55, 5, 1],
    [28, 44, 8, 6],
    [14, 30, 5, 5],
    [15, 16, 3, 1]
])

def make_confusion_matrix():
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure()
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

def calculate_MAE():
    i = np.arange(cm.shape[0]).reshape(-1, 1)
    j = np.arange(cm.shape[1]).reshape(1, -1)
    mae = np.sum(cm * np.abs(i - j)) / np.sum(cm)
    print("MAE:", mae)

if __name__ == '__main__':
    make_confusion_matrix()
    calculate_MAE()
