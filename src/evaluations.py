import json
import os
from sklearn.metrics import confusion_matrix

def compute_metrics(y_true, y_pred):
    """
    Binary classification metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "FPR": fpr,
        "FNR": fnr,
        "TPR": tpr,
        "TNR": tnr
    }


def save_report(model_name, metrics, hyperparams, epochs, datapoints):
    """
    Save experiment report as JSON in /reports
    """

    # project root = parent of src/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reports_dir = os.path.join(project_root, "reports")

    os.makedirs(reports_dir, exist_ok=True)

    report = {
        "model": model_name,
        "metrics": metrics,
        "hyperparameters": hyperparams,
        "epochs": epochs,
        "datapoints": datapoints
    }

    filename = f"{model_name}_report.json"
    filepath = os.path.join(reports_dir, filename)

    with open(filepath, "w") as f:
        json.dump(report, f, indent=4)

    print(f"[OK] Report saved to: {filepath}")
