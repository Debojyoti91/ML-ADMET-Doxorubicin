import pandas as pd
from chembl_webresource_client.new_client import new_client

def get_target_data_for_molecule(chembl_id):
    """
    Fetch target protein data for a given molecule ChEMBL ID.

    Args:
        chembl_id (str): ChEMBL ID of the molecule

    Returns:
        DataFrame: Contains target protein data
    """
    
    # Initialize ChEMBL client
    client = new_client

    # Fetch activities based on molecule ChEMBL ID
    activities = client.activity.filter(molecule_chembl_id=chembl_id).only(['target_chembl_id'])
    
    target_proteins_data = []
    for activity in activities:
        target_id = activity['target_chembl_id']
        target = client.target.filter(target_chembl_id=target_id)
        
        if not target:
            continue

        target_info = target[0]
        keys_to_check = ['target_chembl_id', 'pref_name', 'organism', 'target_type']
        
        # Check if all the required keys are present in the target_info
        if all(key in target_info for key in keys_to_check):
            tax_id = target_info.get('tax_id', None)

            if target_info['organism'] == 'Homo sapiens' and target_info['target_type'] == 'SINGLE PROTEIN':
                protein_data = (
                    target_info['target_chembl_id'], 
                    target_info['pref_name'], 
                    target_info['organism'], 
                    target_info['target_type'], 
                    tax_id
                )
                target_proteins_data.append(protein_data)

    return target_proteins_data


def main():
    chembl_id = 'CHEMBL53463'   #Doxorubicin molecule
    print(f"Fetching data for ChEMBL ID: {chembl_id}")

    # Get target data for the specified molecule
    target_data = get_target_data_for_molecule(chembl_id)
    
    # Convert data to DataFrame
    df = pd.DataFrame(target_data, columns=['ChEMBL ID', 'pref_name', 'organism', 'target_type', 'tax_id'])
    
    # Data cleaning steps
    df = df.drop_duplicates(subset='ChEMBL ID', keep='first')
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    # Save data to a CSV file
    df.to_csv('target_protein_data.csv', index=False)
    print("Saved target protein data to 'target_protein_data.csv'")


if __name__ == "__main__":
    main()

