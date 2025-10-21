# ============================================================
# adme_model.py
# Module for ADME property prediction using SMILES Transformer embeddings
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from transformers import AutoTokenizer, AutoModel


def smiles_embedding(smiles, tokenizer, model, max_length=128):
    """
    Generate mean pooled Transformer embedding for a single SMILES string.
    """
    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def train_adme_model(data_path, out_dir="plots", n_splits=5, n_estimators=100):
    """
    Train and evaluate Random Forest models for multiple ADME endpoints
    using SMILES Transformer embeddings + molecular descriptors.

    Parameters
    ----------
    data_path : str
        Path to the input CSV dataset.
    out_dir : str
        Output directory for saving plots.
    n_splits : int
        Number of folds for K-Fold cross-validation.
    n_estimators : int
        Number of trees in the Random Forest.

    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics (RMSE, MAE, R2) for each target.
    predictions : dict
        Dictionary of cross-validated predictions for each target.
    """

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Step 1: Load Dataset
    # -----------------------------
    df = pd.read_csv(data_path)

    # -----------------------------
    # Step 2: Define Columns
    # -----------------------------
    input_columns = [
        'MolWt', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds',
        'NumAromaticRings', 'NumHeteroatoms', 'FractionCSP3', 'LogP',
        'WienerIndex', 'SURFACE_AREA', 'Water solubility', 'Hydrophilicity'
    ]

    targets = [
        'Permeability', 'Solubility', 'HalfLife', 'Caco2 permeability',
        'Intestinal absorption (human)', 'BBB permeability',
        'CNS permeability', 'Total Clearance',
        'VDss (human)', 'Fraction unbound (human)', 'Skin Permeability'
    ]

    X_numeric = df[input_columns].fillna(df[input_columns].mean())
    y = df[targets].fillna(df[targets].mean())
    smiles_list = df['SMILES'].tolist()

    # -----------------------------
    # Step 3: Load Pretrained Model
    # -----------------------------
    print("Loading pretrained SMILES transformer model...")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    model.eval()

    # -----------------------------
    # Step 4: Generate Embeddings
    # -----------------------------
    print("Generating SMILES embeddings...")
    smiles_embeddings = np.array([
        smiles_embedding(smi, tokenizer, model) for smi in smiles_list
    ])

    # -----------------------------
    # Step 5: Combine Descriptors + Embeddings
    # -----------------------------
    X_combined = np.hstack([X_numeric.values, smiles_embeddings])

    # -----------------------------
    # Step 6: Train and Evaluate
    # -----------------------------
    print("Training and evaluating models with cross-validation...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rf_model = RandomForestRegressor(
        random_state=42, n_estimators=n_estimators, n_jobs=-1
    )

    predictions = {}
    metrics = {}

    for target in targets:
        y_true = y[target].values
        y_pred = cross_val_predict(rf_model, X_combined, y_true, cv=kf, n_jobs=-1)

        predictions[target] = y_pred
        metrics[target] = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

        print(f"{target}: RMSE={metrics[target]['RMSE']:.3f}, "
              f"MAE={metrics[target]['MAE']:.3f}, R²={metrics[target]['R2']:.3f}")

    # -----------------------------
    # Step 7: Visualization
    # -----------------------------
    sns.set_style("white")
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    scatter_color = "#6a0dad"  # Deep purple-blue

    for idx, target in enumerate(targets):
        if target not in predictions:
            continue
        ax = axes[idx]
        ax.scatter(y[target], predictions[target],
                   alpha=0.7, s=70, color=scatter_color, edgecolor='k')
        ax.plot(
            [y[target].min(), y[target].max()],
            [y[target].min(), y[target].max()],
            'r--', lw=2
        )
        ax.set_title(f"{target}\nR²={metrics[target]['R2']:.2f}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Actual', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_axisbelow(False)

    for j in range(len(targets), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "smiles_transformer_adme_predictions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ADME prediction plots to {plot_path}")

    return metrics, predictions

