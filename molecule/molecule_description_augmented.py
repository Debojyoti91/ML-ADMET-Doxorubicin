import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

def calculate_wiener_index(mol):
    """
    Calculate the Wiener index of a molecule.
    
    Args:
    - mol (rdkit.Chem.Mol): Input molecule.
    
    Returns:
    - int: Wiener index.
    """
    num_atoms = mol.GetNumAtoms()
    wiener_index = 0
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            path_length = len(Chem.GetShortestPath(mol, i, j))
            wiener_index += path_length
    return wiener_index // 2

# Load the augmented dataset
dataset = pd.read_csv('augmented_dataset.csv')

# Predefined list of important descriptor names
important_descriptors = [
    "ChEMBL_ID",
    "Name",
    "SMILES",
    "MolWt",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumHeteroatoms",
    "FractionCSP3",
    "LogP",
    "WienerIndex"
]

descriptors_list = []

for idx, row in dataset.iterrows():
    molecule_descriptors = {"ChEMBL_ID": row['ChEMBL ID'], "Name": row['Name'], "SMILES": row['Smiles']}
    molecule_mol = Chem.MolFromSmiles(row['Smiles'])
    
    if molecule_mol is None:
        continue

    for descriptor_name in important_descriptors[3:]:
        if descriptor_name == "LogP":
            descriptor_value = Descriptors.MolLogP(molecule_mol)
        elif descriptor_name == "WienerIndex":
            descriptor_value = calculate_wiener_index(molecule_mol)
        else:
            descriptor_value = getattr(Descriptors, descriptor_name)(molecule_mol)
        molecule_descriptors[descriptor_name] = descriptor_value

    descriptors_list.append(molecule_descriptors)

descriptors_dataset = pd.DataFrame(descriptors_list)
print(descriptors_dataset)

# Using descriptors to calculate ADME properties
df = descriptors_dataset

df['Permeability'] = df['SMILES'].apply(lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x)))
df['Solubility'] = df['SMILES'].apply(lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x)))
df['Metabolism'] = df['SMILES'].apply(lambda x: Descriptors.NumRotatableBonds(Chem.MolFromSmiles(x)))
df['HalfLife'] = df['SMILES'].apply(lambda x: Descriptors.MolMR(Chem.MolFromSmiles(x)))
df['Hydrophilicity'] = df['SMILES'].apply(lambda x: Descriptors.TPSA(Chem.MolFromSmiles(x)))

print(df)

# Save the final dataframe to a CSV
df_adme = pd.DataFrame(df)
df_adme.to_csv('molecule_train_final_dataset.csv', index=False)

