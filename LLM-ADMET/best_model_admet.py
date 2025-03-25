import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from rdkit import Chem
from rdkit.Chem import AllChem

def convert_smiles_to_fingerprint(smiles):
    """Convert SMILES string to ECFP4 fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2)  # ECFP4 fingerprint
    fingerprint_arr = np.zeros((1,))
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_arr)
    return fingerprint_arr

def prepare_data(filename):
    """Load the dataset, generate fingerprints and split into training and testing sets."""
    merged_data = pd.read_csv(filename)
    fingerprint_data = merged_data['SMILES'].apply(convert_smiles_to_fingerprint)
    fp_df = pd.DataFrame(fingerprint_data.tolist(), columns=[f'bit_{i}' for i in range(fingerprint_data.iloc[0].size)])
    merged_data = pd.concat([merged_data, fp_df], axis=1)
    merged_data.drop('SMILES', axis=1, inplace=True)

    # Extracting features and target variables
    features = merged_data.drop(['hERG I inhibitor', 'hERG II inhibitor', 'ChEMBL_ID', 'ChEMBL_ID_prot', 'pref_name_prot'], axis=1)
    target = merged_data['hERG II inhibitor']

    return train_test_split(features, target, test_size=0.2, random_state=42)

def train_and_evaluate(models, param_grids, X_train, y_train, filename="results.txt", best_models_filename="best_models.csv"):
    """Train the given models, perform hyperparameter tuning and save results."""
    best_model = None
    best_score = float('inf')
    
    for model in models:
        print(f"Training {type(model).__name__}...")
        params = param_grids.get(type(model), {})
        grid_search = GridSearchCV(model, params, scoring='accuracy', cv=5)
        grid_search.fit(X_train, y_train)
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = type(model).__name__

        with open(filename, 'a') as file:
            file.write(f"Model: {type(model).__name__}\n")
            file.write(f"Best Accuracy: {grid_search.best_score_}\n")
            file.write(f"Best Parameters: {grid_search.best_params_}\n")
            file.write("======================\n")
    
    with open(best_models_filename, 'w') as file:
        file.write(f"Best Model: {best_model} with Accuracy: {best_score}\n")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data('molecule_protein_merged_data_ADME_tox.csv')

    param_grids = {
        DecisionTreeClassifier: {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        RandomForestClassifier: {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        GradientBoostingClassifier: {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5]
        },
        AdaBoostClassifier: {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1]
        },
        # ... [add other models' param grids as needed]
    }

    models = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        AdaBoostClassifier(),
        # ... [add other models as needed]
    ]

    train_and_evaluate(models, param_grids, X_train, y_train)

