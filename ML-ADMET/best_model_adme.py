import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor

# Load the dataset
df = pd.read_csv('Molecules_adme_updated_train_ml_dataset.csv')

input_columns = [
    'MolWt', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds',
    'NumAromaticRings', 'NumHeteroatoms', 'FractionCSP3', 'LogP', 'WienerIndex',
    'SURFACE_AREA', 'Water solubility', 'Hydrophilicity'
]
inputs = df[input_columns]
outputs = df[['Permeability', 'Solubility', 'Metabolism', 'HalfLife', 'Caco2 permeability', 'Intestinal absorption (human)', 'P-glycoprotein substrate', 'BBB permeability', 'CNS permeability', 'Total Clearance']]
outputs.replace({'Yes': 1, 'No': 0}, inplace=True)
X_train, _, y_train, _ = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Define parameter grids for hyperparameter tuning
param_grids = {
    LinearRegression: {},
    DecisionTreeRegressor: {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    RandomForestRegressor: {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    GradientBoostingRegressor: {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5]
    },
    AdaBoostRegressor: {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1]
    },
    KNeighborsRegressor: {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    LGBMRegressor: {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5]
    }
}

# List of models for training
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    AdaBoostRegressor(),
    KNeighborsRegressor(),
    LGBMRegressor()
]

def train_and_evaluate(models, param_grids, X_train, y_train, outputs, filename="results.txt", best_models_filename="best_models.csv"):
    best_model_for_each_target = {}
    
    for output_column in outputs.columns:
        best_score = float('inf')
        best_model_name = ""
        best_params = None

        for model in models:
            print(f"Training {type(model).__name__} for {output_column}...")
            y_train_single = y_train[output_column]
            params = param_grids.get(type(model), {})
            
            grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5)
            grid_search.fit(X_train, y_train_single)
            
            if -grid_search.best_score_ < best_score:
                best_score = -grid_search.best_score_
                best_model_name = type(model).__name__
                best_params = grid_search.best_params_

            with open(filename, 'a') as file:
                file.write(f"Output: {output_column}, Model: {type(model).__name__}\n")
                file.write(f"Best MSE: {-grid_search.best_score_}\n")
                file.write(f"Best Parameters: {grid_search.best_params_}\n")
                file.write("======================\n")
                
        best_model_for_each_target[output_column] = {
            "model": f"{best_model_name} - {best_params}"
        }

    # Convert best models to DataFrame and save as CSV
    df_best_models = pd.DataFrame({
        "Target": list(best_model_for_each_target.keys()),
        "Best Model with Parameters": [details["model"] for details in best_model_for_each_target.values()]
    })

    df_best_models.to_csv(best_models_filename, index=False)

train_and_evaluate(models, param_grids, X_train, y_train, outputs)

