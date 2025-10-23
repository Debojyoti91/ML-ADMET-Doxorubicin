# ML-ADMET-Doxorubicin

## Project Overview

This repository presents a **unified, explainable machine learning and LLM framework** for predicting and interpreting the pharmacokinetic (ADME) and toxicity (ADMET) properties of **anthracycline derivatives**, with a primary focus on *doxorubicin* and its analogs.  
It combines molecular feature generation, model training, and SHAP-based explainability into a **single modular and interactive workflow**.

The pipeline leverages both **SMILES-based embeddings** and **physicochemical descriptors** to predict properties such as solubility, permeability, clearance, and hERG II inhibition risk. All models are fully interpretable, enabling transparent insights into molecular design and optimization.

---

## 📂 Folder Structure

### 1. `Protein/`
Scripts to retrieve and analyze protein targets relevant to doxorubicin:
- `get_target_protein.py` – Retrieves protein targets using the ChEMBL API.
- `protein_descriptors.py` – Computes sequence- and structure-based descriptors.
- `get_alphafold_3d.py` – Downloads AlphaFold 3D models from UniProt IDs.

### 2. `Molecule/`
Scripts for molecular preparation, descriptor generation, and data augmentation:
- `dox_similar.py` – Identifies structurally similar compounds to doxorubicin.
- `molecule_data_augmentation.py` – Generates functionalized analogs (–NH₂, –OH, –CO, –CH₃).
- `molecule_description_augmented.py` – Computes RDKit and pkCSM-based molecular descriptors.

### 3. `Models/`
Contains the core modeling modules and explainability scripts:
- `adme_model.py` – Regression models for pharmacokinetic property prediction.
- `admet_model.py` – Classification models for toxicity endpoints (e.g., hERG II).
- `shap_pipeline.py` – SHAP-based interpretability pipeline for model explanation.

### 4. `interactive_main.py`
The **main interactive runner** that unifies all modeling steps.  
It handles:
- Data loading and preprocessing  
- ADME regression model training  
- ADMET classification model training  
- SHAP analysis and plot generation  

### 5. `Doxorubicin_interactive.ipynb`
An example notebook demonstrating end-to-end usage of the unified pipeline.  
It provides an interactive environment for training, evaluation, and explainability visualization.

---

## Architecture Overview

### Inputs
- Molecular SMILES strings  
- Calculated RDKit descriptors and pkCSM properties  
- Optionally, pretrained molecular embeddings (e.g., PubChem10M-BPE)

### Pipeline Steps
1. **Data Preparation** — Reads SMILES and computes descriptors.  
2. **ADME Regression** — Predicts continuous properties (e.g., solubility, clearance, half-life).  
3. **ADMET Classification** — Predicts toxicity endpoints such as hERG II inhibition.  
4. **Model Interpretation** — Uses SHAP to identify key molecular features influencing predictions.  
5. **Visualization** — Generates normalized (%SHAP) bar and violin plots for interpretability.

---

##  How It Works

| Step | Description | Key Files / Tools |
|------|--------------|------------------|
| **1. Data Preparation** | Generates descriptors and feature matrices from molecular SMILES | RDKit, pkCSM |
| **2. ADME Modeling** | Trains the LLM model| `adme_model.py` |
| **3. ADMET Modeling** | Classifies molecules by toxicity risk | `admet_model.py` |
| **4. Explainability** | Computes SHAP values and plots feature importance | `shap_pipeline.py` |
| **5. Interactive Execution** | Runs full workflow in one step | `interactive_main.py` |

---

##  Usage Guide

###  Protein Analysis
```bash
# Retrieve target proteins
python Protein/get_target_protein.py

# Generate protein descriptors
python Protein/protein_descriptors.py

# Download AlphaFold 3D structures
python Protein/get_alphafold_3d.py
