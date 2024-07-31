"""
Geneformer HSC Embeddings

This file is used to embed Fibroblast's single cells with Geneformer.

TODO:
1. Set up a single file with all perturbations in a .csv file. Then, when we use an array job i.e. try to run this
   file ~250 times, each instance of running this script will select which specific parameters or perturbations
   to use
2. Set up a system to save the results to a file. This method doesn't need to return anything, but it does need
   to save the perturbed embeddings.
3. It looks to me as if the perturbations occur on line 220 but the embeddings have already occured on line 191. I 
   am not the most familiar with this code, but it does seem unusual/backwards to me.

NOTE: DONE< 7/30
"""

import sys
import seaborn as sns
import pandas as pd 
import numpy as np
from itertools import combinations
from scipy.spatial.distance import squareform, pdist
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torch
import anndata as an
import scanpy as sc
import os
import gc
from importlib import reload

from datasets import Dataset, load_from_disk
from datasets import load_dataset
from geneformer import EmbExtractor

# classifer tools
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


import pandas as pd 
import numpy as np
import anndata as an
import scanpy as sc
import pickle

from datasets import Dataset, load_from_disk, load_dataset
import geneformer

from datetime import datetime

DEFAULT_NAME_PATH = "/nfs/turbo/umms-indikar/shared/projects/geneformer/geneformer/gene_name_id_dict.pkl"
DEFAULT_TOKEN_PATH = "/nfs/turbo/umms-indikar/shared/projects/geneformer/token_dictionary.pkl"
DEFAULT_MEDIAN_PATH = "/nfs/turbo/umms-indikar/shared/projects/geneformer/geneformer/gene_median_dictionary.pkl"

sns.set_style('white')
torch.cuda.empty_cache()



### Parameters v
###

# cells to filter on
initial_cell_type = 'Fibroblast'

# list of genes to perturb with at the front of the list
gene_list = [
    'GATA2', 
    'GFI1B', 
    'FOS', 
    'STAT5A',
    'REL',
    'FOSB',
    'IKZF1',
    'RUNX3',
    'MEF2C',
    'ETV6',
]

############### important! ###############
# added back sampling, now all cells
#num_initial_cells = 10
################  (1/2)  #################
MODEL_PATH = "/scratch/indikar_root/indikar1/shared_data/geneformer/fine_tune/240715_geneformer_cellClassifier_no_induced/ksplit1/"
#     Args:
#         MODEL_PATH (str): Path to the model file.
#         model_type (str, optional): Type of model ('Pretrained' or custom). Default: 'Pretrained'.
#         n_classes (int, optional): Number of output classes for custom models. Default: 0.
#         mode (str, optional): Mode to load the model in ('eval' or 'train'). Default: 'eval'.
model = geneformer.perturber_utils.load_model('Pretrained', 0 , MODEL_PATH, 'eval')


print('model loaded!')


TOKEN_DATA_PATH = "/scratch/indikar_root/indikar1/shared_data/geneformer/resources/token_mapping.csv"
token_df = pd.read_csv(TOKEN_DATA_PATH)


DATA_PATH = "/scratch/indikar_root/indikar1/shared_data/geneformer/fine_tune/hsc.dataset"



###
### Parameters ^


# Uses token_df to translate from gene_list to tokens_list v
def get_tokens_list(gene_list):
    # Get a df of the genes we are perturbing with
    genes = token_df[token_df['gene_name'].isin(gene_list)]
    
    tf_map = dict(zip(genes['gene_name'].values, genes['token_id'].values))
    
    # Create tokens_list by looking up each gene_name in the tf_map
    tokens_list = [tf_map.get(gene_name, gene_name) for gene_name in gene_list]

    return tokens_list


def add_perturbations_to_cell(cell_tokens, perturbation_tokens):
    """
    Requires: cell_tokens is a list of (in our use, 2048) integer tokens, each token representing a gene,
        perturbation_tokens is a list of integer tokens
    Effects: returns final_tokens, a list of tokens with perturbation_tokens at the front
    """
    original_length = len(cell_tokens)

    # Remove existing perturbation tokens from the cell
    cell_tokens = [token for token in cell_tokens if token not in perturbation_tokens]

    # Add perturbations, then slice or pad to match original length
    final_tokens = (perturbation_tokens + cell_tokens)[:original_length]  # Slice if too long
    final_tokens += [0] * (original_length - len(final_tokens))            # Pad if too short

    return final_tokens

def extract_embedding_in_mem(model, data, emb_mode='cell', layer_to_quant=-1, forward_batch_size=10):
    """Extracts embeddings from a model and returns them as a DataFrame.

    This function provides an in-memory extraction of embeddings, allowing for convenient
    manipulation and analysis directly within your Python environment.

    Args:
        model: The model to use for embedding extraction.
        data: The input data for which embeddings need to be extracted.
        emb_mode (str, optional): The embedding mode. Defaults to 'cell'.
        layer_to_quant (int, optional): The layer to quantize. Defaults to -1 (last layer).
        forward_batch_size (int, optional): The batch size for forward passes. Defaults to 10.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted embeddings.

    Raises:
        TypeError: If `model` is not a supported model type.
        ValueError: If `data` is not in the correct format.
    """

    embs = geneformer.emb_extractor.get_embs(
        model,
        data,
        emb_mode,
        layer_to_quant,
        0,  # Assuming this is a constant parameter for the function
        forward_batch_size,
        summary_stat=None,  
        silent=False, 
    )
    data = embs.cpu().numpy()
    if emb_mode=='cell':
        return pd.DataFrame(data)
    else:
        return data


def embedding_to_adata(df: pd.DataFrame, n_dim: int = None) -> an.AnnData:
    """Converts a Pandas DataFrame with an embedding to an AnnData object.

    Args:
        df: The input DataFrame with numerical embedding columns and optional metadata columns.
        n_dim: The number of dimensions to keep in the embedding. If None, all dimensions are kept.

    Returns:
        The converted AnnData object.

    Raises:
        ValueError: If `n_dim` exceeds the available dimensions in the DataFrame.
    """

    if n_dim is not None and n_dim > df.shape[1]:
        raise ValueError(f"n_dim ({n_dim}) exceeds available dimensions ({df.shape[1]})")

    # Assuming embedding columns are those that are not integers
    is_metadata = df.columns.astype(str).str.isdigit()
    metadata_df = df.loc[:, ~is_metadata]
    embedding_df = df.loc[:, is_metadata]

    cell_index = pd.Index([f"C{x}" for x in range(df.shape[0])], name='obs_names')

    if n_dim is not None:
        embedding_df = embedding_df.iloc[:, :n_dim]

    var_index = pd.Index([f"D{x}" for x in range(embedding_df.shape[1])], name='var_names')

    adata = an.AnnData(embedding_df.to_numpy())
    adata.obs_names = cell_index
    adata.var_names = var_index
    adata.obs = metadata_df
    return adata


##########
"""
FOR OUR USE CASE SPECIFICALLY.
"""

def ten_choose_five():
    gene_list = [
    'GATA2', 
    'GFI1B', 
    'FOS', 
    'STAT5A',
    'REL',
    'FOSB',
    'IKZF1',
    'RUNX3',
    'MEF2C',
    'ETV6',
]
# Define the length of sublists
    len_sublist = 5
    
    # Generate all combinations of the specified length
    sublists = list(combinations(gene_list, len_sublist))
    
    # Create the DataFrame
    df = pd.DataFrame({
        'recipe_iteration': range(1, len(sublists) + 1),
        'recipe_list': [list(sublist) for sublist in sublists]
    })
    return df

# Check and ensure all data is of correct type
def check_and_convert(data):
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if not pd.api.types.is_string_dtype(data[col]):
                data[col] = data[col].astype(str)
    return data

    
#######

    
def single_recipe(recipe_list):
    ### Format raw data, print messages to check v
    ###

    # Load from pre-trained data
    raw_data = load_from_disk(DATA_PATH)

    # Convert to DataFrame for filtering
    df = raw_data.to_pandas()
    print("\nOriginal Dataset:")
    print(f"  - Number of samples: {df.shape[0]:,}")
    print(f"  - Number of columns: {df.shape[1]:,}")

    # Filtering
    fb_df = df[df['standardized_cell_type'] == initial_cell_type]

    ############### important! ###############
    # sampling (ADDED BACK!)
    #fb_df = fb_df.sample(num_initial_cells)
    ################  (2/2)  #################

    fb_df = fb_df.reset_index(drop=True)

    # add a cell id
    fb_df['cell_id'] = [f"cell_{i+1}" for i in range(len(fb_df))]
    fb_df['recipe'] = 'raw'  # as opposed to having a speciofic ;-separated recipe list. other entries will have this.
    fb_df['type'] = 'initial' # this dataframe no longer includes 'target'

    print("\nFiltered Dataset:")
    print(f"  - Number of samples: {fb_df.shape[0]:,}")   # Nicer formatting with commas
    print(f"  - Number of columns: {fb_df.shape[1]:,}")

    # Value counts with sorting
    print("\nCell Type Distribution (Filtered):")
    print(fb_df['standardized_cell_type'].value_counts().sort_index())  # Sort for readability

    # Convert back to Dataset
    fb_data = Dataset.from_pandas(fb_df)
    print(f"\nDataset converted back: {fb_data}")

    ###
    ### Format raw data, print messages to check ^

    ############################################################################
    
    ### Perform the perturbation v
    ###

    reprogramming_df = [
        fb_df
    ]

    perturb = fb_df.copy()
    recipe = ";".join(recipe_list)
    perturb['recipe'] = recipe
    perturb['type'] = 'reprogrammed'
    perturb['input_ids'] = perturb['input_ids'].apply(lambda x: add_perturbations_to_cell(x, get_tokens_list(recipe_list)))

    reprogramming_df.append(perturb)

    reprogramming_df = pd.concat(reprogramming_df, ignore_index=True)

    print(f"{reprogramming_df.shape=}")

    ###
    ### Perform the perturbation ^

    ############################################################################
    ### Get the embeddings into an Anndata object v
    ###

    torch.cuda.empty_cache()

    reprogramming_data = Dataset.from_pandas(reprogramming_df)

    reprogramming_embs = extract_embedding_in_mem(
        model, 
        reprogramming_data, 
        layer_to_quant=-1,
        forward_batch_size=100,
    )
    print(f"{reprogramming_embs.shape=}")

    # translate into an anndata object and plot
    reprogramming_adata = embedding_to_adata(reprogramming_embs)
    reprogramming_adata.obs = reprogramming_df.copy()


    reprogramming_adata.obs.head()

    return reprogramming_adata



def main():

    jobNumber = int(sys.argv[1])
    print(jobNumber)


    # For our iterations specifically...
    all_input_recipes = ten_choose_five()
    recipe_list = all_input_recipes.at[all_input_recipes[all_input_recipes['recipe_iteration'] == jobNumber].index[0], 'recipe_list']


    if torch.cuda.is_available(): 
        print("CUDA is available! Devices: ", torch.cuda.device_count()) 
        print("Current CUDA device: ", torch.cuda.current_device()) 
        print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device())) 
    else: print("CUDA is not available")

    # run on a specific perturbation ( last thing not yet: change with ten_choose_five() )
    reprogramming_adata = single_recipe(recipe_list)


    reprogramming_adata.obs = check_and_convert(reprogramming_adata.obs)
    reprogramming_adata.var = check_and_convert(reprogramming_adata.var)
   
        
    filepath = f"/nfs/turbo/umms-indikar/shared/projects/geneformer/fib15k/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_job_number_{jobNumber}.h5ad"

    try:
        reprogramming_adata.write(filepath)
        print(f"File successfully written to {filepath}")
    except Exception as e:
        print(f"Error occurred: {e}")
    





# These lines of code will call the main function when someone uses
# ```
# python geneformerHSCembeddings.py
# ```
if __name__ == "__main__":
    main()


