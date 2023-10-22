import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops

def load_dataset(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def perform_functional_group_operations(mol):
    """
    Apply functional group addition/removal on the given molecule.
    
    Args:
    - mol (rdkit.Chem.Mol): An RDKit molecule object.
    
    Returns:
    - list: A list of new augmented RDKit molecule objects.
    """
    new_mols = []
    fg_patterns = ['[NH2]', '[OH]', '[CO]', '[CH3]']
    for pattern in fg_patterns:
        substruct = Chem.MolFromSmarts(pattern)
        matches = mol.GetSubstructMatches(substruct)
        for match in matches:
            new_mol = Chem.RWMol(mol)
            atom_indices = match
            if isinstance(atom_indices, int):
                atom_indices = [atom_indices]
            for atom_idx in atom_indices:
                atom = new_mol.GetAtomWithIdx(atom_idx)
                if atom.GetDegree() > 0:
                    atom.SetAtomicNum(1)
            new_mols.append(new_mol)
        
        new_mol = Chem.RWMol(mol)
        new_mol = Chem.rdmolops.DeleteSubstructs(new_mol, substruct)
        new_mols.append(new_mol)
    return new_mols

def augment_data(df):
    """
    Augment the given dataframe with additional molecular data.
    
    Args:
    - df (pd.DataFrame): Input dataframe.
    
    Returns:
    - pd.DataFrame: Augmented dataframe.
    """
    augmented_data = []
    valid_smiles = set()
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['Smiles'])
        if mol:
            valid_smiles.add(row['Smiles'])
            augmented_mols = perform_functional_group_operations(mol)
            for new_mol in augmented_mols:
                augmented_smile = Chem.MolToSmiles(new_mol)
                if augmented_smile not in valid_smiles:
                    valid_smiles.add(augmented_smile)
                    augmented_data.append((row['ChEMBL ID'], row['Name'], augmented_smile))
    
    augmented_df = pd.DataFrame(augmented_data, columns=['ChEMBL ID', 'Name', 'Smiles'])
    return pd.concat([df, augmented_df]).drop_duplicates(subset='Smiles').reset_index(drop=True)

def save_dataset(df, file_path):
    """Save the dataframe to a CSV file."""
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    # Load the original dataset
    df = load_dataset('Doxorubicin_similar_compounds.csv')
    
    # Perform data augmentation
    augmented_df = augment_data(df)
    
    # Save the augmented dataset
    save_dataset(augmented_df, 'augmented_dataset.csv')

