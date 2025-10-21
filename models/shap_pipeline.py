# shap_pipeline.py
# Comprehensive ADME and ADMET SHAP analysis and visualization pipeline

import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def run_shap_pipeline(base_dir="/content/drive/MyDrive/Doxorubicin_ADMET"):
    """
    Run the full SHAP workflow for ADME and ADMET models.
    This function trains Random Forest models, computes SHAP values,
    prints top-5 descriptor importance summaries, and generates
    normalized pastel bar plots for visualization.

    Parameters
    ----------
    base_dir : str
        Base directory containing 'data' and 'plots' folders.
    """

    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------
    # ADME section (Regression)
    # -------------------------------
    data_adme = os.path.join(data_dir, "molecule_ADME_ML_train_data_final.csv")

    input_columns = [
        'MolWt', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds',
        'NumAromaticRings', 'NumHeteroatoms', 'FractionCSP3', 'LogP',
        'WienerIndex', 'SURFACE_AREA', 'Water solubility', 'Hydrophilicity'
    ]

    adme_targets = [
        'Permeability', 'Solubility', 'HalfLife', 'Caco2 permeability',
        'Intestinal absorption (human)', 'BBB permeability',
        'CNS permeability', 'Total Clearance',
        'VDss (human)', 'Fraction unbound (human)', 'Skin Permeability'
    ]

    adme_df = pd.read_csv(data_adme)
    X_adme = adme_df[input_columns]
    Y_adme = adme_df[adme_targets]

    print("\nStarting ADME SHAP analysis")
    adme_summary = {}
    adme_shap = {}

    for target in adme_targets:
        y = Y_adme[target].values
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_adme, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_adme)
        adme_shap[target] = shap_values

        mean_abs = np.abs(shap_values).mean(axis=0)
        vals_norm = (mean_abs / np.sum(mean_abs)) * 100
        top_idx = np.argsort(vals_norm)[::-1][:5]
        adme_summary[target] = list(zip(np.array(input_columns)[top_idx], vals_norm[top_idx]))

        print(f"Computed SHAP values for {target}")

    np.savez_compressed(os.path.join(out_dir, "shap_values_adme_all.npz"), **adme_shap)
    np.save(os.path.join(out_dir, "feature_names_adme.npy"), np.array(input_columns))
    print(f"Saved ADME SHAP arrays and feature names in {out_dir}")

    # -------------------------------
    # ADMET section (Classification)
    # -------------------------------
    data_admet = os.path.join(data_dir, "molecule_ADME_tox_train_data_final.csv")
    df = pd.read_csv(data_admet)
    print(f"\nLoaded ADMET dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace({'Yes': 1, 'No': 0})

    def smiles_to_ecfp4(smiles, n_bits=2048, radius=2):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return np.zeros((n_bits,), dtype=int)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    print("Generating molecular fingerprints")
    fps = df["SMILES"].apply(smiles_to_ecfp4)
    fp_df = pd.DataFrame(fps.tolist(), index=df.index)
    fp_df.columns = [f'bit_{i}' for i in range(fp_df.shape[1])]
    df = pd.concat([df, fp_df], axis=1)

    drop_cols = ['hERG I inhibitor', 'ChEMBL_ID', 'Name', 'SMILES']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    X_admet = df.drop(columns=['hERG II inhibitor'], errors='ignore')
    y_admet = df['hERG II inhibitor'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_admet, y_admet, test_size=0.2, random_state=42, stratify=y_admet
    )

    rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("Trained Random Forest model for ADMET")

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_admet)
    np.savez_compressed(os.path.join(out_dir, "shap_values_admet.npz"), shap_values_admet=shap_values)
    np.save(os.path.join(out_dir, "feature_names_admet.npy"), np.array(list(X_admet.columns)))
    print("Computed and saved ADMET SHAP arrays")

    sv = shap_values[1] if isinstance(shap_values, (list, tuple)) else shap_values
    mean_abs = np.abs(sv).mean(axis=0)
    feature_names = np.array(list(X_admet.columns))

    desc_cols = [
        'MolWt', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds',
        'NumAromaticRings', 'NumHeteroatoms', 'FractionCSP3', 'LogP',
        'WienerIndex', 'SURFACE_AREA', 'Water solubility', 'Hydrophilicity'
    ]
    desc_present = [d for d in desc_cols if d in feature_names]
    desc_vals = [float(np.mean(mean_abs[np.where(feature_names == d)])) for d in desc_present]

    desc_vals = np.array(desc_vals)
    desc_norm = (desc_vals / np.sum(mean_abs)) * 100
    top_idx = np.argsort(desc_norm)[::-1][:5]
    admet_summary = list(zip(np.array(desc_present)[top_idx], desc_norm[top_idx]))

    # -------------------------------
    # Summary printout
    # -------------------------------
    print("\nNormalized Top-5 Descriptor Importance Profiles")
    for target, pairs in adme_summary.items():
        print(f"\nTop-5 descriptors for ADME endpoint: {target}")
        for name, val in pairs:
            print(f"{name:<30s} {val:>10.3f} %")

    print("\nTop-5 descriptors for ADMET (hERG II Inhibition)")
    for name, val in admet_summary:
        print(f"{name:<30s} {val:>10.3f} %")

    # -------------------------------
    # Visualization
    # -------------------------------
    adme_npz = np.load(os.path.join(out_dir, "shap_values_adme_all.npz"), allow_pickle=True)
    admet_npz = np.load(os.path.join(out_dir, "shap_values_admet.npz"), allow_pickle=True)
    feat_names = np.load(os.path.join(out_dir, "feature_names_adme.npy"), allow_pickle=True)

    adme_top5 = {}
    for target in adme_targets:
        if target not in adme_npz.files:
            continue
        sv = np.asarray(adme_npz[target], dtype=float)
        mean_abs = np.abs(sv).mean(axis=0).flatten()
        top_idx = np.argsort(mean_abs)[::-1][:5]
        names = [str(x) for x in np.ravel(feat_names[top_idx])]
        vals = mean_abs[top_idx].flatten()
        vals_norm = (vals / np.sum(mean_abs)) * 100
        adme_top5[target] = (names, vals_norm)

    admet_sv = admet_npz['shap_values_admet']
    if isinstance(admet_sv, (list, tuple)):
        admet_sv = admet_sv[1]
    admet_sv = np.asarray(admet_sv, dtype=float)
    admet_mean = np.abs(admet_sv).mean(axis=0).flatten()
    admet_names = np.load(os.path.join(out_dir, "feature_names_admet.npy"), allow_pickle=True)
    mask = [d for d in admet_names if d in desc_cols]
    idx = [list(admet_names).index(m) for m in mask]
    vals = admet_mean[idx].flatten()
    top_idx = np.argsort(vals)[::-1][:5]
    names = [str(x) for x in np.array(mask)[top_idx]]
    vals = vals[top_idx]
    vals_norm = (vals / np.sum(admet_mean)) * 100
    admet_top5 = (names, vals_norm)

    n_adme = len(adme_top5)
    ncols = 2
    nrows = int(np.ceil((n_adme + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 22))
    axes = axes.flatten()
    pastel_palette = sns.color_palette("pastel")

    for i, (target, (names, vals)) in enumerate(adme_top5.items()):
        colors = pastel_palette[:len(names)]
        axes[i].bar(names, vals, color=colors, edgecolor='gray')
        axes[i].set_title(target, fontsize=11)
        axes[i].set_ylabel('% Relative SHAP Importance', fontsize=10)
        axes[i].tick_params(axis='x', labelsize=9)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        axes[i].set_ylim(0, max(vals) * 1.2)

    ax = axes[len(adme_top5)]
    names, vals = admet_top5
    colors = pastel_palette[:len(names)]
    ax.bar(names, vals, color=colors, edgecolor='gray')
    ax.set_title('hERG II Inhibition (ADMET)', fontsize=11)
    ax.set_ylabel('% Relative SHAP Importance', fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, max(vals) * 1.2)

    for j in range(len(adme_top5) + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle(
        'Top-5 Normalized Molecular Descriptor Importance Profiles for ADME and ADMET Models',
        fontsize=20, y=1.02
    )
    plot_path = os.path.join(out_dir, "SHAP_pastel_normalized_vertical_bars_all_endpoints.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nSHAP analysis completed. Figure saved to {plot_path}")


if __name__ == "__main__":
    run_shap_pipeline()

