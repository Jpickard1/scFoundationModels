"""
This file is for embedding single cell data using a variety of foundation models. It is packed as one file
to provide a standard interface
"""

## General
import re
import os
import gzip
import sys
import numpy as np
import pandas as pd
import sklearn
import warnings
from tqdm import tqdm
import argparse
import pickle
import torch
print('general imports')

# Bioinformatics
import scanpy as sc
import anndata as ad
print('bioinf imports')

## scgpt
try:
    from pathlib import Path
    from scipy.stats import mode
    import scgpt as scg
    import faiss
    print('scgpt will work')
except:
    print('scgpt wont work')


## tGPT
try:
    from torch.utils.data import DataLoader, Dataset
    from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2Model
    from tGPTutils import LineDataset, get_top_genes_per_cell
    print('tGPT will work')
except:
    print('tGPT wont work')


## geneformer
try:
    import scipy.sparse as sp
    from datasets import Dataset, load_from_disk
    sys.path.append('/home/jpic/geneformer_dev/scripts')
    import geneformer_utils as gtu
    from gfUtils import *
    print('geneformer will work')
except:
    print('geneformer wont work')



def embed(adata, model, model_directory=None, output_directory='/nfs/turbo/umms-indikar/shared/projects/foundation_models/experiments', output_file=None, verbose=False):
    """
    This file embeds anndata and writes them to a .h5ad file on Turbo. embed.log was created in the default output directory to indicate what each generated file is.
    Params:
    ------
        adata: ann data object
        model: name of model to use
    """
    base_name, input_file, base_dir = None, None, None
    if isinstance(adata, str):
        # Get file path
        input_file = adata
        base_name  = os.path.splitext(os.path.basename(input_file))[0]
        base_dir   = os.path.splitext(os.path.dirname(input_file))[0]

        # Load ann data object
        adata = sc.read_h5ad(input_file)

        if output_directory is None:
            output_directory = base_dir

    if output_file is None:
        if base_name is not None:
            output_file = base_name + '_' + model + '.h5ad'
        else:
            output_file = model + '.h5ad'

    #if not output_file.endswith('.h5ad'):
    #    output_file += '.h5ad'


    if output_directory is None:
        # default spot to save embeddings
        output_directory = '~/.cache/embeddings/'
        full_path = os.path.expanduser(directory_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(full_path, exist_ok=True)

    if model is None:
        print('model must exist')
        return

    # Perform embedding    
    if model == 'scgpt':
        embed_adata = scgptEmbed(adata, model_directory)
    elif model == 'geneformer':
        embed_adata = geneformerEmbed(adata, output_path = output_directory, filename=output_file, verbose=verbose)
    elif model == 'tGPT':
        embed_adata = tGPTembed(adata)
    else:
        print('We need a valid model')
        return
    # Free cuda!
    torch.cuda.empty_cache()
    
    # Add in some metadata
    embed_adata.obs = adata.obs
    
    return embed_adata

def scgptEmbed(adata, model_directory):
    """
    This function embeds AnnData with scGPT
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 14, 2024
    model_dir   = Path("/nfs/turbo/umms-indikar/shared/projects/foundation_models/scGPT_human")
    embed_adata = scg.tasks.embed_data(
        adata,            # data
        model_directory,  # pointer at model
        gene_col    = "gene_name", # the index of the adata.var are genes/tokens for scGPT
        obs_to_save = list(adata.obs.columns),  # save all of the metadata in the new dataframe
        batch_size  = 64,      # I am not sure what this does
        return_new_adata=True,
    )
    return embed_adata

def tGPTembed(adata):
    """
    This function embeds AnnData with tGPT
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 15, 2024
    topGenes = get_top_genes_per_cell(adata)

    ds = LineDataset(topGenes)
    dl = DataLoader(ds, batch_size=64)
    
    Xs = []

    # load some tGPT params
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(device)
    tokenizer_file = "/nfs/turbo/umms-indikar/shared/projects/foundation_models/transcriptome-gpt-1024-8-16-64" 
    checkpoint     = "/nfs/turbo/umms-indikar/shared/projects/foundation_models/transcriptome-gpt-1024-8-16-64" ## Pretrained model
    celltype_path  = "/nfs/turbo/umms-indikar/shared/projects/foundation_models/example_inputs/tGPT/Muris_cell_labels.txt.gz" ## Cell type annotation
    max_len        = 64 ## Number of top genes used for analysis
    text_file      = "/nfs/turbo/umms-indikar/shared/projects/foundation_models/example_inputs/tGPT/Muris_gene_rankings.txt.gz"  ## Gene symbols ranked by exprssion
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_file)
    print('Tokenizer set')
    model = GPT2LMHeadModel.from_pretrained(checkpoint,output_hidden_states = True).transformer
    print('model set')
    model = model.to(device)
    print('model to device')
    model.eval()
    print('model eval')

    for a in tqdm(dl, total=len(dl)):
        batch = tokenizer(a, max_length= max_len, truncation=True, padding=True, return_tensors="pt")
    
        for k, v in batch.items():
            batch[k] = v.to(device)
    
        with torch.no_grad():
            x = model(**batch)
        
        eos_idxs = batch.attention_mask.sum(dim=1) - 1
        xx = x.last_hidden_state
           
        result_list = [[] for i in range(len(xx))]
    
        for j, item in enumerate(xx):
            result_list[j] = item[1:int(eos_idxs[j]),:].mean(dim =0).tolist()
            
        Xs.extend(result_list)
        
    features = np.stack(Xs)

    emebddingAdata = ad.AnnData(X=features)
    return emebddingAdata

def geneformerEmbed(adata, output_path=None, filename=None, verbose=False):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 14, 2024
    # from gfUtils import *

    output_path = os.path.join(output_path, filename + '.dataset')
    
    # Default values
    MODEL_PATH          = "/nfs/turbo/umms-indikar/shared/projects/geneformer/geneformer-12L-30M/"
    DEFAULT_NAME_PATH   = "/nfs/turbo/umms-indikar/shared/projects/geneformer/geneformer/gene_name_id_dict.pkl"
    DEFAULT_TOKEN_PATH  = "/nfs/turbo/umms-indikar/shared/projects/geneformer/token_dictionary.pkl"
    DEFAULT_MEDIAN_PATH = "/nfs/turbo/umms-indikar/shared/projects/geneformer/geneformer/gene_median_dictionary.pkl"
    MODEL_INPUT_SIZE    = 2048
    NUMBER_PROC         = 16
    TARGET_SUM          = 10000
    GENE_ID             = 'ensembl_id'
    COUNTS_COLUMN       = 'n_counts'
    LAYER               = 'X'
    GENE_NAME_COLUMN    = 'gene_name'

    # set values used for embedding
    global model_size
    token_path            = DEFAULT_TOKEN_PATH
    median_path           = DEFAULT_MEDIAN_PATH
    n_proc                = NUMBER_PROC
    model_size            = MODEL_INPUT_SIZE
    target_sum            = TARGET_SUM
    gene_id               = GENE_ID
    aggregate_transcripts = False
    counts_column         = COUNTS_COLUMN
    layer                 = LAYER
    gene_names            = DEFAULT_NAME_PATH
    gene_name_column      = GENE_NAME_COLUMN
    map_names             = False
    num_cells             = None # all cells, useful for testing 

    torch.cuda.empty_cache()
    
    ###########################################
    #
    #   TOKENIZE COUNTS DATA FOR GENEFORMER
    #
    ###########################################
    if not os.path.exists(output_path):
        print('Tokenizing the dataset')
        print("Loading gene tokenization data...") if verbose else None
        gene_token_dict, gene_keys, genelist_dict = load_gene_tokenization(token_path)
        print(f"Loaded {len(gene_token_dict)} gene tokens") if verbose else None
        
        print("Loading gene median expression data...") if verbose else None
        gene_median_dict = load_gene_median_dict(median_path)
        print(f"Loaded {len(gene_median_dict)} gene median expression values") if verbose else None
        
        if map_names:
            print("Loading gene name mapping data...") if verbose else None
            gene_names = load_gene_names(gene_names)
            print(f"Loaded {len(gene_names)} gene name mappings") if verbose else None
        
        print(f"Using AnnData with shape {adata.shape}") if verbose else None
        
        if map_names:
            print("Mapping gene names to Ensembl IDs...") if verbose else None
            adata = map_gene_names(adata, gene_id, gene_name_column, gene_names)
        
        if not layer == 'X':
            print(f"Using layer '{layer}' for expression data...") if verbose else None
            adata.X = adata.layers[layer]
            
        print("Checking for and/or calculating total counts per cell...") if verbose else None
        adata = check_counts_column(adata, counts_column)
        
        # Tokenize and rank genes
        print("Tokenizing and ranking genes...") if verbose else None
        tokenized_cells, cell_metadata = tokenize_anndata(
            adata, genelist_dict, gene_median_dict,
            target_sum=target_sum, gene_id=gene_id, counts_column=counts_column,
            gene_token_dict=gene_token_dict
        )
        print(f"Processed {len(tokenized_cells)} cells") if verbose else None
        
        # Create Hugging Face dataset
        print("Creating Hugging Face dataset...") if verbose else None
        dataset_dict = {
            "input_ids": tokenized_cells,
            **cell_metadata
        }
        output_dataset = Dataset.from_dict(dataset_dict)
        print(f"Dataset has {len(output_dataset)} examples") if verbose else None
        
        # Format cell features
        print("Formatting cell features...") if verbose else None
        dataset = output_dataset.map(format_cell_features, num_proc=n_proc)
        
        # Save dataset
        print(f"Saving processed dataset to {output_path}...") if verbose else None
        
        save_hf_dataset(dataset, output_path, overwrite=True)
        print("Processing completed successfully!") if verbose else None
    else:
        print('Dataset is pretokenized')
        
    ###########################################
    #
    #   EMBED TOKENS WITH GENEFORMER TO ANNDATA
    #
    ###########################################
    dataset_path = output_path
    
    print(MODEL_PATH)
    
    print(f"Loading model from '{MODEL_PATH}'...") if verbose else None
    model = gtu.load_model(MODEL_PATH)
    print("Model loaded successfully!") if verbose else None
    
    print(f"Loading dataset from '{dataset_path}' (up to {num_cells} cells)...") if verbose else None
    try:
        df = gtu.load_data_as_dataframe(dataset_path, num_cells=num_cells)
        data = Dataset.from_pandas(df)
        df = df.drop(columns='input_ids')
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset_path}'") if verbose else None
        sys.exit(1)
    except Exception as e:  # Catching other potential errors
        print(f"Error loading dataset: {e}") if verbose else None
        sys.exit(1)
    print("Dataset loaded successfully!") if verbose else None
    
    print("Extracting embeddings...") if verbose else None
    embs = gtu.extract_embedding_in_mem(model, data)
    adata = gtu.embedding_to_adata(embs)
    adata.obs = df.astype(str).reset_index().copy()
    print("Embeddings extracted successfully!") if verbose else None

    return adata

    #print(f"Writing results to '{outpath}'...") if verbose else None
    #try:
    #    adata.write(outpath)
    #except Exception as e:
    #    print(f"Error writing output file: {e}") if verbose else None
    #    sys.exit(1)
    #print("Output file written successfully!") if verbose else None
    #sys.exit(0)





