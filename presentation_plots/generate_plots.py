# generate_plots.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model

# ----------------------------
# Paths
# ----------------------------
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
PLOTS_DIR = os.path.join(PROJECT_DIR, "presentation_plots")
SAVED_MODELS_DIR = os.path.join(PROJECT_DIR, "saved_models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------
# Load test data
# ----------------------------
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# ----------------------------
# Load reports
# ----------------------------
reports = {}
for file in os.listdir(REPORTS_DIR):
    if file.endswith(".json"):
        with open(os.path.join(REPORTS_DIR, file)) as f:
            reports[file.replace("_report.json","")] = json.load(f)

model_names = list(reports.keys())
print("Loaded models:", model_names)

# ----------------------------
# 1️⃣ Accuracy bar plot
# ----------------------------
acc_values = []
acc_labels = []
for m in model_names:
    acc = reports[m].get("accuracy", None)
    if acc is not None:
        acc_values.append(acc)
        acc_labels.append(m)

plt.figure(figsize=(10,5))
sns.barplot(x=acc_labels, y=acc_values, palette="Set2")
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_per_model.png"), dpi=300)
plt.close()

# ----------------------------
# 2️⃣ Performance metrics (TPR, TNR, FPR, FNR)
# ----------------------------
metrics = ["TPR","TNR","FPR","FNR"]
plt.figure(figsize=(12,6))
x = np.arange(len(model_names))
width = 0.18

for i, metric in enumerate(metrics):
    vals = []
    labels = []
    for m in model_names:
        if metric in reports[m]:
            vals.append(reports[m][metric])
            labels.append(m)
    if vals:
        plt.bar(x[:len(vals)] + i*width, vals, width, label=metric)

if vals:
    plt.xticks(x[:len(labels)] + 1.5*width, labels, rotation=45)
    plt.ylabel("Metric Value")
    plt.title("Performance Metrics per Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "performance_metrics.png"), dpi=300)
plt.close()

# ----------------------------
# 3️⃣ Training vs Validation Loss for DL models
# ----------------------------
for model in model_names:
    history_file = os.path.join(REPORTS_DIR, f"{model}_history.npy")
    if os.path.exists(history_file):
        history = np.load(history_file, allow_pickle=True).item()
        plt.figure(figsize=(8,4))
        plt.plot(history.get("loss", []), label="Train Loss")
        plt.plot(history.get("val_loss", []), label="Val Loss")
        plt.title(f"{model} Training vs Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{model}_loss.png"), dpi=300)
        plt.close()

# ----------------------------
# 4️⃣ ROC curves (only models with y_prob)
# ----------------------------
plt.figure(figsize=(8,6))
for model in model_names:
    y_prob_file = os.path.join(REPORTS_DIR, f"{model}_y_prob.npy")
    if os.path.exists(y_prob_file):
        y_prob = np.load(y_prob_file)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=300)
plt.close()

# ----------------------------
# 5️⃣ Precision-Recall curves
# ----------------------------
plt.figure(figsize=(8,6))
for model in model_names:
    y_prob_file = os.path.join(REPORTS_DIR, f"{model}_y_prob.npy")
    if os.path.exists(y_prob_file):
        y_prob = np.load(y_prob_file)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, label=model)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_curves.png"), dpi=300)
plt.close()

# ----------------------------
# 6️⃣ GRU Embeddings PCA (if exists)
# ----------------------------
gru_encoder_path = os.path.join(SAVED_MODELS_DIR, "gru_encoder.keras")
if os.path.exists(gru_encoder_path):
    from tensorflow.keras import models
    gru_encoder_model = load_model(gru_encoder_path, compile=False)
    encoder = models.Model(inputs=gru_encoder_model.input,
                           outputs=gru_encoder_model.get_layer("embedding").output)
    X_test_seq = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    embeddings = encoder.predict(X_test_seq, verbose=0)
    from sklearn.decomposition import PCA
    emb_2d = PCA(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    plt.scatter(emb_2d[:,0], emb_2d[:,1], c=y_test, cmap="coolwarm", edgecolor='k', alpha=0.7)
    plt.title("GRU Embeddings PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Class (0=Malignant,1=Benign)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gru_embeddings.png"), dpi=300)
    plt.close()

print("✅ All available presentation plots generated in 'presentation_plots' folder.")
