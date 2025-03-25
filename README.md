# ML-ADMET-Doxorubicin

## 🧬 Project Overview

This repository presents a unified computational pipeline for discovering optimized anthracycline derivatives with improved pharmacokinetics and reduced toxicity. Centered on doxorubicin and its stereoisomers, this pipeline integrates **target-specific protein retrieval**, **structure-based ligand design**, **large-scale docking**, **molecular dynamics simulations**, and **machine learning–based ADME/ADMET prediction**.

By systematically expanding the chemical space of anthracyclines and combining ligand-based modeling with biological target interactions, this repository facilitates the discovery of novel drug candidates with reduced cardiotoxicity and enhanced metabolic profiles.

---

## 📂 Folder Structure

### 1. `Protein/`
Scripts to retrieve and analyze human protein targets relevant to doxorubicin:

- `get_target_protein.py` – Retrieves target proteins using ChEMBL API.
- `protein_descriptors.py` – Computes protein-specific descriptors for modeling.
- `get_alphafold_3d.py` – Downloads AlphaFold 3D structures using UniProt IDs.

### 2. `Molecule/`
Scripts to process Doxorubicin analogs, generate new derivatives, and compute molecular descriptors:

- `dox_similar.py` – Extracts structurally similar compounds from ChEMBL.
- `molecule_data_augmentation.py` – Performs functional group modifications (–NH₂, –OH, –CO, –CH₃) for chemical diversity.
- `molecule_description_augmented.py` – Calculates molecular descriptors and ADME estimates using RDKit and pkCSM.

### 3. `ML-ADME/`
Scripts for training machine learning models to predict pharmacokinetics (ADME) and toxicity (ADMET):

- `best_model_adme.py` – Trains regression models on ADME endpoints using a hybrid feature set (RDKit + LLM embeddings).
- `best_model_admet.py` – Implements classification models for hERG II inhibition risk prediction.

---

## 🧪 Pipeline Summary

### 🔬 Step 1: Protein Target Retrieval & Representation
- Retrieve 201 **human-derived single-chain proteins** from ChEMBL.
- Extract structural models from **AlphaFold** for docking and descriptor generation.

### 🧪 Step 2: Ligand Design & Data Augmentation
- Start with **doxorubicin and its 19 stereoisomers**.
- Generate 222 diverse ligands using targeted functional group modifications.
- Compute descriptors (logP, TPSA, H-bond donors/acceptors, rotatable bonds, etc.).

### 🔄 Step 3: Docking & Molecular Dynamics
- Perform **2,375 docking simulations** using AutoDock Vina (ligands × proteins).
- Use **GROMACS** for 50 ns MD simulations on top ligand-protein complexes.
- Analyze RMSD, RMSF, radius of gyration, SASA, and binding energy fluctuations.

### 🤖 Step 4: Machine Learning Predictions
- **ADME Regression**:
  - Predict properties like solubility, permeability, clearance, half-life.
  - Integrate **SMILES embeddings** from pretrained transformer models (PubChem10M SMILES-BPE).
  - Train Random Forest models using hybrid features.

- **ADMET Classification**:
  - Focus on **hERG II inhibition** as cardiotoxicity marker.
  - Train supervised models (Random Forest, XGBoost, Decision Trees).
  - Achieved **>92% accuracy** with balanced precision and recall.

---



## 🧰 Usage Guide

### 🔧 Protein Analysis
```bash
# Fetch target proteins
python Protein/get_target_protein.py

# Generate protein descriptors
python Protein/protein_descriptors.py

# Download 3D structures via AlphaFold
python Protein/get_alphafold_3d.py
coding standards and are thoroughly documented.



# Find similar compounds to Doxorubicin
python Molecule/dox_similar.py

# Apply functional group modifications
python Molecule/molecule_data_augmentation.py

# Compute descriptors and ADME features
python Molecule/molecule_description_augmented.py



# ADME property prediction (regression)
python ML-ADME/best_model_adme.py

# ADMET toxicity classification (hERG II)
python ML-ADME/best_model_admet.py


🤝 Contributions

Contributions are welcome! You may:

Add new predictive models (e.g., for liver toxicity or bioavailability).
Integrate molecular generation (e.g., using reinforcement learning).
Optimize hyperparameters or extend LLM embeddings.
Improve documentation or pipeline modularity.
Please follow standard coding practices and provide comments for reproducibility.



