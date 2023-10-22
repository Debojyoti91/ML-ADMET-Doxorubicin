import requests
import xml.etree.ElementTree as ET
import pandas as pd

class ChEMBLSimilarity:

    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/"

    @staticmethod
    def fetch_similar_compounds(smiles, threshold=90):
        """Retrieve compounds similar to the provided SMILES string from ChEMBL."""
        url = ChEMBLSimilarity.BASE_URL + f"similarity/{smiles}/{threshold}"
        response = requests.get(url)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        compounds = [{
            'ChEMBL ID': molecule.findtext('.//molecule_chembl_id'),
            'Smiles': molecule.findtext('.//canonical_smiles')
        } for molecule in root.findall('.//molecule') if molecule.findtext('.//canonical_smiles') is not None]

        return compounds

    @staticmethod
    def fetch_molecule_name(chembl_id):
        """Retrieve the preferred name for a given ChEMBL ID."""
        url = ChEMBLSimilarity.BASE_URL + f"molecule/{chembl_id}"
        response = requests.get(url)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        return root.findtext('.//pref_name') or "N/A"


def main():
    doxorubicin_smiles = 'COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)C[C@@](O)(C(=O)CO)C[C@@H]3O[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1'

    try:
        compounds = ChEMBLSimilarity.fetch_similar_compounds(doxorubicin_smiles)
        for compound in compounds:
            compound['Name'] = ChEMBLSimilarity.fetch_molecule_name(compound['ChEMBL ID'])

        df = pd.DataFrame(compounds)
        df.dropna(subset=['Smiles'], inplace=True)
        df.to_csv('Doxorubicin_similar_compounds.csv', index=False)
        print(df)
    except requests.RequestException as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

