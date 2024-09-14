"""
allTFperturbations.py

This file is designed to run in a noninteractive way to generate a series of new .h5ad files each with a perturbation to a single transcription factor. Nat wrote 99% of all the code in here, and Joshua packaged in as a .py file so that it could run over shabbos / the weekend.

Auth: Joshua Pickard
      jpic@umich.edu
Date: September 13, 2024
"""
import numpy as np
import pandas as pd
import scanpy as sp
import anndata as ad # JP add this line
import os

def translate_ids(list):
    # Extract base IDs from list_b by removing version numbers
    new_list = [id.split('.')[0] for id in list]
    return new_list

def validate_ensembl_tfs(tf_ensid_list, counts_ensid_list):
    missing_tfs = []
    for TF in tf_ensid_list:
        if TF not in counts_ensid_list:
            missing_tfs.append(TF)
    
    if missing_tfs:
        print("TFs not found in counts_ensid_list:")
        for tf in missing_tfs:
            print(tf)
        return False
    
    return True

def iterate_perturb_id_counts(adata, tf_list, scalar_list):
    """
    Applies perturbations to the expression data of specified transcription factors across multiple scalars 
    and stores the resulting AnnData objects in a dictionary.

    This function performs the following steps:
    1. Iterates over a list of scalar values.
    2. For each scalar, creates a copy of the AnnData object to preserve the original data.
    3. Applies the `perturb_counts` function to scale the expression data of genes listed in `tf_list` by
       the maximum gene expression of each cell and the current scalar.
    4. Stores the perturbed AnnData object in a dictionary with the scalar as the key.

    Parameters:
    tf_list (list): A list of gene symbols (transcription factors) to be perturbed.
    scalar_list (list): A list of scalar values for scaling the gene expression.
    adata (AnnData): The AnnData object containing gene expression data (cells x genes).

    Returns:
    dict: A dictionary where keys are scalar values and values are the corresponding perturbed AnnData objects.
    """
    
    adata_dict = {}
    
    for scalar in scalar_list:
        # Create a copy of the AnnData object for each scalar value
        adata_temp = adata.copy()
        
        # Apply perturb_counts to the copied AnnData object
        perturbed_adata = perturb_id_counts(adata_temp, tf_list, scalar) # JP change this line
        
        # Store the perturbed AnnData object in the dictionary with scalar as the key
        adata_dict[scalar] = perturbed_adata
    
    return adata_dict

def perturb_id_counts(adata, tf_list, scalar): 
    """
    Applies a perturbation to the expression data of specific genes in an AnnData object.

    This function performs the following steps:
    1. Computes the maximum gene expression level for each cell.
    2. Applies a scaling operation to the expression levels of genes listed in `tf_list`.
       - Each entry of these genes in the matrix is multiplied by the maximum expression level 
         of its respective cell and a specified scalar value.
    3. Updates the AnnData object with new columns:
       - 'scaled': A boolean column indicating whether each gene is in the `tf_list`.
       - 'scaled_by': Contains the scaling factor used for each gene (the product of the maximum 
         expression level of each cell and the scalar), or `1` if the gene was not in `tf_list`.
    
    Parameters:
    tf_list (list): A list of gene symbols to be perturbed.
    scalar (float): The scalar value used to scale the expression levels.
    adata (AnnData): The AnnData object containing gene expression data.

    Returns:
    AnnData: The updated AnnData object with applied perturbations and new columns.
    """

    # Create a boolean mask for genes in tf_list
    gene_mask = adata.var['ensemblid'].isin(tf_list)
    
    # Save the original state of the parameter objects, in case some tfs do not translate (failsafe)
    original_X = adata.X.copy()
    original_gene_mask = gene_mask.copy()
    
    # Compute maximum expression level of each cell
    max_exp = np.max(adata.X, axis=1)

    """This is new today. v """
    # Raise an error if any of the gene names in tf_list do not match column names (we will manually update these in adata):
    missing_genes = [gene for gene in tf_list if gene not in adata.var['ensemblid'].values]
    
    if missing_genes:
        # Restore original parameter objects
        adata.X = original_X
        gene_mask = original_gene_mask
        raise ValueError(f"Genes {missing_genes} not found in anndata object")

    else:    
        
        # Apply the scaling operation to the specified genes
        adata.X[:, gene_mask] = max_exp * scalar
        
        # Add/Update 'scaled' column in var
        adata.var['scaled'] = gene_mask
        
        # Add/Update 'scaled_by' column in var
        adata.var['scaled_by'] = scalar  # Default value for genes not in tf_list

        # Add/Update 'scaled_by' column in var
        adata.obs['U'] = scalar  # Default value for genes not in tf_list
    # adata.var = 
    return adata

# Load reprogramming recipes
df = pd.read_csv('../notebooks/reprogramming/data/HumanTFs_v_1.01.csv') # JP this line is changed
df.head()

# Load firboblast source cells
DATAPATH = "/nfs/turbo/umms-indikar/shared/projects/DARPA_AI/in-silico-reprogramming/unperturbed"
FILE = "fibroblast.h5ad"
adata = sp.read_h5ad(os.path.join(DATAPATH, FILE))

adata.var['ensemblid'] = adata.var['ensemblid'].str.split('.').str[0] # JP Add this line

print(adata.var.head())


#ensembl ids from the counts matrix
counts_ensid_list = adata.var['ensemblid'].values.tolist()

# there were version numbers ending with " .'#' " that needed to be removed
counts_ensid_list = translate_ids(counts_ensid_list)


# ensembl ids from the transcription factor list, which in this case is also the perturbation list (testing one at a time)
tf_ensid_list = df['Ensembl ID'].values.tolist()



validate_ensembl_tfs(tf_ensid_list, counts_ensid_list)

# when we perturb each tf in tf_ensid_list, we are just going to skip the one problematic one
problem_ens_ids = ['ZNF73_HUMAN', 'ENSG00000204828', 'DUX1_HUMAN', 'DUX3_HUMAN', 'ENSG00000262156', 'ENSG00000196101']


print(len(tf_ensid_list))
print(len(problem_ens_ids))
tf_ensid_list = [id for id in tf_ensid_list if id not in problem_ens_ids]
print(len(tf_ensid_list))

output_directory = "/nfs/turbo/umms-indikar/shared/projects/DARPA_AI/in-silico-reprogramming/all-tfs"
scalars = [0.5, 0.75, 1.001]

# Filter tf_ensid_list to remove problematic Ensembl IDs
tf_ensid_list = [id for id in tf_ensid_list if id not in problem_ens_ids]
print(f"Total Ensembl IDs to process: {len(tf_ensid_list)}")

# Clean Ensembl IDs in adata.var['ensemblid'] to remove anything after '.'
cleaned_ensembl_ids = adata.var['ensemblid'].str.split('.').str[0]

for i in range(len(tf_ensid_list)):
    try:
        print(f"\nProcessing index: {i}")
        val = df['Ensembl ID'].iloc[i]
        TFs = val.split()  # Split in case there are multiple TFs
        print(f"Current TFs: {TFs}")

        # Check if all TFs are in the cleaned Ensembl IDs
        missing_in_adata = [tf for tf in TFs if tf not in cleaned_ensembl_ids.values]
        if missing_in_adata:
            print(f"Skipping missing genes: {missing_in_adata}")
            continue  # Skip this set of TFs if any are missing

        # Validate TFs with counts_ensid_list
        if validate_ensembl_tfs(tf_ensid_list, counts_ensid_list):
            TFs_str = "_".join(TFs)
            output_path = os.path.join(output_directory, f"{TFs_str}.h5ad")

            # Check if the file already exists
            if os.path.exists(output_path):
                print(f"{output_path} already exists: continuing to next TF!")
                continue

            print(f"Perturbing TFs: {TFs}")
            adataDict = iterate_perturb_id_counts(adata.copy(), TFs, scalars)

            # Concatenate all AnnData objects along the observations axis
            concatenated_adata = ad.concat(list(adataDict.values()), axis=0)

            # Copy var information from the original AnnData
            concatenated_adata.var = adata.var.copy()
            
            # Save the concatenated AnnData object to the file
            concatenated_adata.write_h5ad(output_path)
            print("    File created successfully")
        else:
            print(f"The file made from {val} could not be created.")
    except Exception as e:
        print(f"Error encountered at index {i}: {e}")

print('All recipes complete!')
