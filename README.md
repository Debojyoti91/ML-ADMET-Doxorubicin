**Repository: ML-ADMET-Doxorubicin**

**Description:** This repository is dedicated to the comprehensive assessment of the ADMET properties of Doxorubicin, as well as molecules structurally analogous to it. The tools herein facilitate the systematic retrieval of data pertaining to singular protein targets originating exclusively from human sources. Furthermore, these utilities enable the computation of intricate molecular descriptors, laying the foundation for the deployment of advanced machine learning models. These models are meticulously designed to predict the intricate behavior of the drug and its analogs within biological matrices.

**Folder Structure:**
1. **Protein**

   Contains scripts related to fetching and analyzing target proteins.

   - **get_target_protein.py:** Fetches target protein data based on a given molecule's ChEMBL ID.   
   - **protein_descriptors.py:** Calculates specific molecular descriptors for protein sequences.   
   - **get_alphafold_3d.py:** Facilitates the download of AlphaFold models for sets of UniProt IDs.   

2. **Molecule**

   Contains scripts related to fetching similar compounds to Doxorubicin, performing data augmentation, and predicting molecular descriptors.

   - **dox_similar.py:** Fetches compounds similar to Doxorubicin from the ChEMBL database.   
   - **molecule_data_augmentation.py:** Augments a dataset containing molecular structures.   
   - **molecule_description_augmented.py:** Calculates molecular descriptors and apparent ADME properties for the augmented dataset.   

3. **ML-ADME**

   Scripts aimed at training machine learning models to predict ADME properties based on molecular descriptors.

   - **best_model_adme.py:** Trains multiple regression models to predict ADME properties.   
   - **best_model_admet.py:** Predicts the hERG II inhibitor property using classification models.   

**Usage:**
- **Protein Data Extraction and Analysis:** Start by running **get_target_protein.py** to fetch target protein data related to Doxorubicin. Use **protein_descriptors.py** to calculate molecular descriptors for the fetched protein sequences. To get 3D structures of the proteins, use **get_alphafold_3d.py**.
- **Molecular Data Handling:** To find compounds similar to Doxorubicin, run the **dox_similar.py** script. Use **molecule_data_augmentation.py** to augment the dataset of molecular structures. **molecule_description_augmented.py** will help in calculating molecular descriptors and predicting ADME properties for the augmented dataset.
- **Machine Learning Predictions:** For regression tasks related to ADME properties, use **best_model_adme.py**. For classification tasks, particularly for predicting the hERG II inhibitor property, utilize the **best_model_admet.py** script. Results generated from the machine learning scripts can be found in **results.txt** and **best_models.csv**.

**Contributions:** Feel free to contribute to this repository by adding new scripts, improving existing ones, or expanding the range of ADMET properties that can be predicted. Please ensure that all contributions adhere to established coding standards and are thoroughly documented.

