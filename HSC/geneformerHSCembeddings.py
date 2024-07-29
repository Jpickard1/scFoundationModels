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
import geneformer as gtu

# classifer tools
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# local imports
sys.path.insert(0, '../../scripts/')
import geneformer_utils as gtu

sns.set_style('white')
torch.cuda.empty_cache()

# Uses token_df to translate from gene_list to tokens_list v
def get_tokens_list(gene_list):
    """
    TODO: <Please write doc strings to describe the inputs and outputs of this method>
    """
    # Get a df of the genes we are perturbing with
    genes = token_df[token_df['gene_name'].isin(gene_list)]

    tf_map = dict(zip(genes['gene_name'].values, genes['token_id'].values))

    # Create tokens_list by looking up each gene_name in the tf_map
    tokens_list = [tf_map.get(gene_name, gene_name) for gene_name in gene_list]

    return tokens_list

def add_perturbations_to_cell(cell_tokens, perturbation_tokens):
    """
    TODO: <Please write doc strings to describe the inputs and outputs of this method>
    """
    original_length = len(cell_tokens)

    # Remove existing perturbation tokens from the cell
    cell_tokens = [token for token in cell_tokens if token not in perturbation_tokens]

    # Add perturbations, then slice or pad to match original length
    final_tokens = (perturbation_tokens + cell_tokens)[:original_length]  # Slice if too long
    final_tokens += [0] * (original_length - len(final_tokens))            # Pad if too short

    return final_tokens

def main():
    """
    TODO: <Please write doc strings to describe the inputs and outputs of this method and how it works>
    """
    ###########################################################################
    ### Parameters v
    ###
    ###    TODO: Please explain what goes on in this sections. I know it looks
    ###          super simple, but pretend someone who knows less than you will
    ###          be responsible for running and modifying this code in the
    ###          future.

    jobNumber = sys.argv[1]

    # TODO: Get list of genes for the specific job number
    # perturbations = pd.read_csv(we can make a file of all the perturbations)
    # gene_list = perturbations[:, jobNum]

    # cells to filter on
    initial_cell_type = 'Fibroblast'

    # list of genes to perturb with at the front of the list
    # TODO: remove this list of genes once the above TODO is complete
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
    
    model_path = "/scratch/indikar_root/indikar1/shared_data/geneformer/fine_tune/240715_geneformer_cellClassifier_no_induced/ksplit1/"

    token_data_path = "/scratch/indikar_root/indikar1/shared_data/geneformer/resources/token_mapping.csv"

    data_path = "/scratch/indikar_root/indikar1/shared_data/geneformer/fine_tune/hsc.dataset"

    ###
    ### Parameters ^

    ###########################################################################

    ### Preliminaries v
    ###
    ###    TODO: Please explain what goes on in this sections.

    if torch.cuda.is_available(): 
        print("CUDA is available! Devices: ", torch.cuda.device_count()) 
        print("Current CUDA device: ", torch.cuda.current_device()) 
        print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device())) 
    else: print("CUDA is not available")

    model = gtu.load_model(model_path)
    print('model loaded!')

    token_df = pd.read_csv(token_data_path)
    token_df.head()

    ###
    ### Preliminaries ^

    ###########################################################################
    
    ### Format raw data, print messages to check v
    ###

    # Load from pre-trained data
    raw_data = load_from_disk(data_path)

    # Convert to DataFrame for filtering
    df = raw_data.to_pandas()
    print("\nOriginal Dataset:")
    print(f"  - Number of samples: {df.shape[0]:,}")
    print(f"  - Number of columns: {df.shape[1]:,}")

    # Filtering
    fb_df = df[df['standardized_cell_type'] == initial_cell_type]

    # add a cell id
    fb_df['cell_id'] = [f"cell_{i+1}" for i in range(len(fb_df))]
    fb_df['recipe'] = 'raw'  # as opposed to having a speciofic ;-separated recipe list. other entries will have this.
    fb_df['type'] = 'initial' # this dataframe no longer includes 'target'

    # display some information about the dataset
    print("\nFiltered Dataset:")
    print(f"  - Number of samples: {fb_df.shape[0]:,}")   # Nicer formatting with commas
    print(f"  - Number of columns: {fb_df.shape[1]:,}")

    # Value counts with sorting
    print("\nCell Type Distribution (Filtered):")
    print(fb_df['standardized_cell_type'].value_counts().sort_index())  # Sort for readability

    # Construct the dataset
    fb_data = Dataset.from_pandas(fb_df)
    print(f"\nDataset converted back: {fb_data}")

    ###
    ### Format raw data, print messages to check ^

    ###########################################################################

    ### Get the embeddings into an Anndata object v
    ###

    reload(gtu) # TODO: I think we should delete this line - JP
    torch.cuda.empty_cache()

    # Construct the embeddings of the dataset
    fb_embs = gtu.extract_embedding_in_mem(
        model, 
        fb_data, 
        layer_to_quant=-1,
        forward_batch_size=100,
    )
    print(f"{fb_embs.shape=}")

    # translate into an anndata object
    fb_adata = gtu.embedding_to_adata(fb_embs)
    fb_adata.obs = fb_df.copy()
    fb_adata.obs.head() # TODO: I think we could delete this line - JP

    ###
    ### Get the embeddings into an Anndata object ^

    ###########################################################################

    ### Perform the perturbation v
    ###

    reprogramming_df = [
        fb_df
    ]

    perturb = fb_df.copy()
    recipe = ";".join(gene_list)
    perturb['recipe'] = recipe
    perturb['type'] = 'reprogrammed'
    perturb['input_ids'] = perturb['input_ids'].apply(lambda x: add_perturbations_to_cell(x, get_tokens_list(gene_list)))

    reprogramming_df.append(perturb)

    reprogramming_df = pd.concat(reprogramming_df, ignore_index=True)

    print(f"{reprogramming_df.shape=}")
    reprogramming_df.sample(10) # TODO: I think we should delete this line - JP

    ###
    ### Perform the perturbation ^

    ###########################################################################

    # TODO: Save the reprogrammed dataframe to a file.



# These lines of code will call the main function when someone uses
# ```
# python geneformerHSCembeddings.py
# ```
if __name__ == "__main__":
    main()


