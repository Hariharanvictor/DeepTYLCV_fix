import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import math

# === Read CSV ===
file_path = "/home/vinoth/Hari_proj/TYLCV/webserver/Github_code/DeepTYLCV/prediction_results_severity_the_latest_12_12.csv"
df = pd.read_csv(file_path)

# === Prepare data ===
probs = df['Probability'].astype(float).values
targets = np.array([1 if 'Severe' in x else 0 for x in df['ID']])

# === Define metric functions ===
def calculate_metrics(probs, targets, threshold=0.5):
    predictions = np.where(probs >= threshold, 1, 0)
    tp = ((predictions == 1) & (targets == 1)).sum()
    tn = ((predictions == 0) & (targets == 0)).sum()
    fp = ((predictions == 1) & (targets == 0)).sum()
    fn = ((predictions == 0) & (targets == 1)).sum()

    def safe_div(x, y): return x / y if y != 0 else 0

    sn = safe_div(tp, tp + fn)
    sp = safe_div(tn, tn + fp)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    bacc = (sn + sp) / 2
    prec = safe_div(tp, tp + fp)
    f1 = safe_div(2 * prec * sn, prec + sn)
    mcc = safe_div((tp * tn - fp * fn),
                   math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) != 0 else 0
    auc_roc = roc_auc_score(targets, probs)

    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Sensitivity (SEN)': sn, 'Specificity (SPE)': sp,
        'Accuracy': acc, 'Balanced Accuracy (BACC)': bacc,
        'F1 Score': f1, 'MCC': mcc, 'AUC': auc_roc
    }

# === Compute and print metrics ===
metrics = calculate_metrics(probs, targets)
print(metrics)
