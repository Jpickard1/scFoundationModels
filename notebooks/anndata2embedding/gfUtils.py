"""
Much of this code comes from the geneformer publication as well as the geneformer_dev repository
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
import scanpy as sc
import anndata as an
from datasets import Dataset, load_from_disk
import torch


def check_counts_column(adata, counts_column):
    """Checks for and calculates a total counts column in AnnData.

    This function examines the AnnData object's observation (`obs`) columns for the specified 
    `counts_column`. If it doesn't exist, the function calculates the sum of each row (cell) 
    across all features in the data matrix (`X`) and stores it as a new column in `obs`.

    Args:
        adata: An AnnData object containing the data to be analyzed.
        counts_column: A string representing the desired name for the total counts column.

    Returns:
        adata: The modified AnnData object, now with the `counts_column` present (either 
               pre-existing or newly calculated).
    """
    obs_columns = adata.obs.columns
    
    if counts_column in obs_columns:
        return adata
    else:
        adata.obs[counts_column] = adata.X.sum(axis=1)
        return adata
    
    
def map_gene_names(adata, gene_id, gene_name_column, gene_names):
    """A function mapping gene names to gene ids """
    var_columns = adata.var.columns
    
    if gene_id in var_columns:
        return adata
    else:
        adata.var[gene_id] = adata.var[gene_name_column].map(gene_names)
        return adata
    
    
def load_gene_names(gene_names_file):
    """
    Loads a gene median dictionary from a pickle file.

    Args:
        gene_names_file (str): Path to the pickle file containing the gene names dictionary.

    Returns:
        dict: A dictionary mapping gene names to IDs
    """

    with open(gene_names_file, "rb") as f:
        gene_names_dict = pickle.load(f)

    return gene_names_dict


def load_gene_median_dict(gene_median_file):
    """
    Loads a gene median dictionary from a pickle file.

    Args:
        gene_median_file (str): Path to the pickle file containing the gene median dictionary.

    Returns:
        dict: A dictionary mapping gene IDs to their median expression values.
    """

    with open(gene_median_file, "rb") as f:
        gene_median_dict = pickle.load(f)

    return gene_median_dict


def load_gene_tokenization(token_dictionary_file):
    """
    Loads gene tokenization data from a pickle file.

    Args:
        token_dictionary_file (str): Path to the pickle file containing the gene-token dictionary.

    Returns:
        dict: Gene-token dictionary (Ensembl ID: token).
        list: List of all gene keys (Ensembl IDs).
        dict: Dictionary mapping gene keys to True (used for selecting genes later).
    """

    with open(token_dictionary_file, "rb") as f:
        gene_token_dict = pickle.load(f)

    gene_keys = list(gene_token_dict.keys())

    # Optimization: Pre-allocate the list for slight performance improvement
    genelist_dict = dict.fromkeys(gene_keys, True)

    return gene_token_dict, gene_keys, genelist_dict


def rank_genes(gene_vector, gene_tokens):
    """Ranks genes based on expression values in descending order.

    Args:
        gene_vector (numpy.ndarray): Array of gene expression values.
        gene_tokens (numpy.ndarray): Array of corresponding gene tokens.

    Returns:
        numpy.ndarray: Array of gene tokens sorted by descending expression value.
    """
    return gene_tokens[np.argsort(-gene_vector)]


def normalize_counts(adata_chunk,  counts_column='n_counts', target_sum=10000):
    """Normalizes gene expression counts within a chunk of AnnData.

    Args:
        adata_chunk (AnnData): A chunk of the AnnData object containing gene expression data.
        counts_column (str): Name of the column in `adata_chunk.obs` containing the total counts per cell.
        target_sum (float): The desired total count per cell after normalization.
        norm_factor_vector (numpy.ndarray): An array of normalization factors for each gene.

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix containing the normalized gene expression counts.

    This function performs the following steps:
        1. Extracts the total counts per cell from the specified column (`counts_column`).
        2. Normalizes the gene expression matrix (`adata_chunk.X`) by dividing by the total counts 
           and multiplying by the `target_sum`.
        3. Further adjusts the normalized values by dividing by the gene-specific normalization 
           factors (`norm_factor_vector`).
        4. Returns the normalized expression matrix as a sparse CSR matrix for efficient storage 
           and computation.
    """
    
    n_counts = adata_chunk.obs[counts_column].values[:, None]  # Cell counts as column vector
    X_norm = adata_chunk.X / n_counts * target_sum / norm_factor_vector
    return sp.csr_matrix(X_norm)  # Efficient sparse representation


def tokenize_anndata(adata, genelist_dict, gene_median_dict, 
                     chunk_size=100000, target_sum=10000, 
                     counts_column='n_counts', gene_id="ensembl_id", gene_token_dict=None):
    """
    Tokenizes and ranks genes within an AnnData object, optimizing for memory efficiency.

    This function processes gene expression data in chunks, applies normalization, and ranks genes
    for each cell based on their expression levels. The resulting tokenized and ranked gene
    representations, along with cell metadata, are returned.

    Args:
        adata (AnnData): The AnnData object containing gene expression data.
        genelist_dict (dict): Dictionary mapping gene IDs to boolean values indicating relevance.
        gene_median_dict (dict): Dictionary mapping gene IDs to their median expression values.
        chunk_size (int, optional): Number of cells to process in each chunk (default: 1000).
        target_sum (int, optional): Target sum for count normalization (default: 10000).
        counts_column (str, optional): The column in `adata.obs` containing cell counts (default: 'n_counts').
        gene_id (str, optional): The column in `adata.var` containing gene IDs (default: 'ensembl_id').

    Returns:
        tuple: 
            - list: List of tokenized and ranked gene lists for each cell.
            - dict: Dictionary containing cell metadata (keys are metadata column names).
    """
    # Filter relevant miRNAs
    coding_miRNA_mask = np.array([genelist_dict.get(i, False) for i in adata.var[gene_id]])
    coding_miRNA_loc = np.where(coding_miRNA_mask)[0]

    # Extract miRNA information
    coding_miRNA_ids = adata.var[gene_id].iloc[coding_miRNA_loc]
    norm_factor_vector = np.array([gene_median_dict[i] for i in coding_miRNA_ids])
    coding_miRNA_tokens = np.array([gene_token_dict[i] for i in coding_miRNA_ids])

    tokenized_cells = []
    file_cell_metadata = {k: [] for k in adata.obs.columns}  # Initialize metadata dict

    # Process in chunks for memory efficiency
    for chunk_start in range(0, adata.shape[0], chunk_size):
        chunk_end = chunk_start + chunk_size
        adata_chunk = adata[chunk_start:chunk_end, coding_miRNA_loc]
        
        # Normalize counts (could be replaced with the untested function above)
        n_counts = adata_chunk.obs[counts_column].values[:, None]
        X_norm = adata_chunk.X / n_counts * target_sum / norm_factor_vector
        X_norm = sp.csr_matrix(X_norm)  

        # Tokenize and rank genes for each cell in chunk
        for i in range(X_norm.shape[0]):
            ranks = rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices])
            ranks = list(ranks[~np.isnan(ranks)].astype(int))

            tokenized_cells.append(ranks)

        # Update metadata
        for k in adata.obs.columns:
            file_cell_metadata[k].extend(adata_chunk.obs[k].astype(str).tolist())

    return tokenized_cells, file_cell_metadata


def format_cell_features(example):
    """
    Truncates gene tokens (`input_ids`) to `model_size` and adds a `length` feature.

    Args:
        example (dict): Cell data with `input_ids` (list of gene tokens).

    Returns:
        dict: Modified cell data with truncated `input_ids` and added `length`.
    """
    MODEL_INPUT_SIZE    = 2048    # this value is hardcoded
    model_size = MODEL_INPUT_SIZE
    example["input_ids"] = example["input_ids"][0:model_size] 
    example["length"] = len(example["input_ids"]) 
    return example


def save_hf_dataset(dataset: Dataset, output_path: str, overwrite=True):
    """
    Saves a Hugging Face Dataset to disk at a specified file path.

    This function serializes a Hugging Face `Dataset` object and saves it to disk in the Arrow format.

    Args:
        dataset (Dataset): The Hugging Face `Dataset` object to be saved.
        output_path (str): The full file path (including the filename) where the dataset will be saved. 
        overwrite (bool, optional): If `True`, an existing dataset at `output_path` will be overwritten. 
                                   If `False` and the file exists, a `FileExistsError` is raised (default: True).

    Raises:
        TypeError: If `dataset` is not a Hugging Face `Dataset` instance.
        FileExistsError: If `output_path` points to an existing file and `overwrite` is False.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("The provided dataset is not a Hugging Face Dataset.")

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Dataset '{output_path}' already exists. Set `overwrite=True` to overwrite."
        )
    dataset.save_to_disk(output_path)
