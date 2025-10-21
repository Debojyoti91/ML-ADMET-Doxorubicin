# ============================================================
# ADMET Multi-Model Classifier (Colab Ready Standalone)
# ============================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem

# ------------------------------------------------------------
# File paths (update if needed)
# ------------------------------------------------------------
DATA_PATH = "/content/drive/MyDrive/Doxorubicin_ADMET/data/molecule_protein_merged_data_ADME_tox.csv"
OUT_DIR = "/content/drive/MyDrive/Doxorubicin_ADMET/plots"
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# SMILES → ECFP4 fingerprint conversion
# ------------------------------------------------------------
def convert_smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ------------------------------------------------------------
# Load and preprocess dataset
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset → {df.shape[0]} rows × {df.shape[1]} columns")

# Encode Yes/No → 1/0
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].replace({"Yes": 1, "No": 0})

# Generate molecular fingerprints
print("Generating ECFP4 fingerprints...")
fps = df["SMILES"].apply(convert_smiles_to_fingerprint)
fp_df = pd.DataFrame(fps.tolist(), index=df.index)
fp_df.columns = [f"bit_{i}" for i in range(fp_df.shape[1])]
df = pd.concat([df, fp_df], axis=1)

# Drop irrelevant/non-numeric columns
drop_cols = [
    "hERG I inhibitor",
    "ChEMBL_ID",
    "SMILES",
    "ChEMBL_ID_prot",
    "pref_name_prot",
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

# Features and target
X = df.drop(columns=["hERG II inhibitor"], errors="ignore")
y = df["hERG II inhibitor"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Final feature matrix: {X.shape}, target vector: {y.shape}")
print(f"Target distribution: {np.bincount(y)}")

# ------------------------------------------------------------
# Define all models
# ------------------------------------------------------------
models = [
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
    ("MLP_100", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
    ("MLP_50", MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)),
    ("MLP_200", MLPClassifier(hidden_layer_sizes=(200,), max_iter=500, random_state=42)),
    ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
]

# Add voting ensemble
voting_clf = VotingClassifier(estimators=models, voting="soft", n_jobs=-1)
models.append(("Voting", voting_clf))

# ------------------------------------------------------------
# 5-Fold Cross-Validation
# ------------------------------------------------------------
kf = KFold(n_splits=5, random_state=42, shuffle=True)
metrics = {}
best_model = None
best_acc = 0.0

print("\nPerforming 5-fold cross-validation...")
for name, model in models:
    try:
        scores = cross_val_score(model, X, y, scoring="accuracy", cv=kf, n_jobs=-1)
        mean_acc = scores.mean()
        metrics[name] = mean_acc
        print(f"{name:<15s} → mean CV accuracy = {mean_acc:.4f}")

        if mean_acc > best_acc:
            best_model = model
            best_acc = mean_acc
    except Exception as e:
        print(f"{name:<15s} → failed ({e})")

# ------------------------------------------------------------
# Final training & evaluation on test set
# ------------------------------------------------------------
if best_model is not None:
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    print(f"\nBest model: {type(best_model).__name__}")
    print(f"Test Accuracy = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall = {rec:.4f}")
    print(f"ROC-AUC = {auc:.4f}")

    # --------------------------------------------------------
    # Visualization: bar plot of actual vs predicted
    # --------------------------------------------------------
    plt.figure(figsize=(9, 6))
    df_plot = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    sns.countplot(data=pd.melt(df_plot), x="value", hue="variable", palette="Set3")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(f"Actual vs Predicted ({type(best_model).__name__})")
    plt.legend(title="Data", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "admet_barplot.png"), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nBar plot saved to {OUT_DIR}/admet_barplot.png")

else:
    print("All models failed — please check input data types.")

