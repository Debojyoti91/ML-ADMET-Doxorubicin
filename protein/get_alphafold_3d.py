import pandas as pd
import os

def download_alphafold_model(uniprot_ids, target_directory='alphafold_results', database_version='v2'):
    """
    Download AlphaFold models for a list of UniProt IDs.

    Args:
        uniprot_ids (list): List of UniProt IDs to fetch AlphaFold models for.
        target_directory (str, optional): Directory to save the downloaded models. Defaults to 'alphafold_results'.
        database_version (str, optional): Version of the AlphaFold database to fetch models from. Defaults to 'v2'.
    """
    
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    for uniprot_id in uniprot_ids:
        # Generate the AlphaFold model URL
        model_url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{database_version}.pdb'
        
        # Define the file path for the downloaded model
        model_filepath = os.path.join(target_directory, f'{uniprot_id}.pdb')
        
        # Download the AlphaFold model file using 'curl'
        os.system(f'curl {model_url} -o {model_filepath}')
        
        # Verify the download
        if os.path.isfile(model_filepath):
            print(f'Downloaded AlphaFold model file for UniProt ID {uniprot_id} and saved as {model_filepath}')
        else:
            print(f'Failed to download AlphaFold model file for UniProt ID {uniprot_id}')


def main():
    # Load data from CSV
    df = pd.read_csv('target_protein_data.csv')

    # Extract the UniProt IDs
    uniprot_ids = df['uniprot_id']

    # Initiate the download process
    download_alphafold_model(uniprot_ids)


if __name__ == "__main__":
    main()

