import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def calculate_descriptors(protein_sequence):
    """
    Calculate the descriptors for a given protein sequence.

    Args:
        protein_sequence (str): Protein sequence.

    Returns:
        tuple: Descriptors for the protein sequence.
    """
    
    mol = Chem.MolFromFASTA(protein_sequence)
    
    if mol is None:
        return None, None, None, None, None, None

    num_hbond_donors = rdMolDescriptors.CalcNumHBD(mol)
    num_hbond_acceptors = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    return num_hbond_donors, num_hbond_acceptors, tpsa, num_rotatable_bonds, num_aromatic_rings, fraction_csp3


def main():
    # Load the data
    df = pd.read_csv("target_protein_data.csv")

    # Calculate the descriptors for each protein sequence and unpack them into respective columns
    descriptors = ['num_hbond_donors', 'num_hbond_acceptors', 'TPSA', 'num_rotatable_bonds', 'num_aromatic_rings', 'FractionCSP3']
    df[descriptors] = df['sequence'].apply(calculate_descriptors).apply(pd.Series)

    # Optionally drop the sequence column if not needed
    df = df.drop('sequence', axis=1)

    # Save the enriched data back to CSV
    df.to_csv('target_protein_descriptor.csv', index=False)

    print("Saved descriptors to 'target_protein_descriptor.csv'")


if __name__ == "__main__":
    main()

