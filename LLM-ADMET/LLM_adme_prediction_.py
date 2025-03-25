
"""
ADME Property Prediction using SMILES Transformer Embeddings and Molecular Descriptors

This script:
1. Loads ADME-related molecular data.
2. Extracts traditional descriptors and generates SMILES embeddings using a pretrained Transformer.
3. Combines descriptors and embeddings for model training.
4. Applies 5-fold cross-validation using Random Forest Regressor for multiple targets.
5. Evaluates and visualizes performance for each target.

Author: Your Name
"""

# Step 1: Import Libraries
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

# Step 2: Load Dataset
# Replace this path with a local or relative path for public release
data_path = 'path/to/Molecules_adme_updated_train_ml_dataset.csv'
df = pd.read_csv(data_path)

# Step 3: Define Feature Columns
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

# Step 4: Load Pretrained SMILES Transformer Model
print("Loading pretrained transformer model...")
tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
model.eval()

# Step 5: Generate SMILES Embeddings
def smiles_embedding(smiles):
    inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

print("Generating SMILES embeddings...")
smiles_embeddings = np.array([smiles_embedding(smi) for smi in smiles_list])

# Step 6: Combine Descriptors and Embeddings
X_combined = np.hstack([X_numeric.values, smiles_embeddings])

# Step 7: Train and Evaluate Model using 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_model = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1)

predictions = {}
metrics = {}

print("Training and evaluating models...")
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

    print(f"{target}: RMSE={metrics[target]['RMSE']:.3f}, MSE={metrics[target]['MSE']:.3f}, "
          f"MAE={metrics[target]['MAE']:.3f}, R²={metrics[target]['R2']:.3f}")

# Step 8: Visualization of Predictions
sns.set_style("white")
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

scatter_color = "#6a0dad"  # Deep purple-blue

for idx, target in enumerate(targets):
    ax = axes[idx]
    ax.scatter(y[target], predictions[target], alpha=0.7, s=70, color=scatter_color, edgecolor='k')
    ax.plot([y[target].min(), y[target].max()], [y[target].min(), y[target].max()], 'r--', lw=2)
    ax.set_title(f"{target}\nR²={metrics[target]['R2']:.2f}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_axisbelow(False)

for j in range(len(targets), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "smiles_transformer_adme_predictions.png"), dpi=300, bbox_inches="tight")
plt.show()
