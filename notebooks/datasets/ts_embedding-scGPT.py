# ts_embedding_1.py
import os
import sys
import scanpy as sc
import anndata as ad

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mode
import sklearn
import warnings
import scgpt as scg
import faiss

# sys.path.append('../anndata2embedding')
# from embed import *

def main():
    job_number = int(sys.argv[1])
    print('Start')
    print(job_number)
    
    DATAPATH = "/nfs/turbo/umms-indikar/shared/projects/DGC/data/tabula_sapiens/extract/"
    OUTPUTPATH = '/nfs/turbo/umms-indikar/shared/projects/foundation_models/experiments/tabulaSapiens'
    h5ad_files = []
    for root, dirs, files in os.walk(DATAPATH):
        for file in files:
            if file.endswith('.h5ad'):
                h5ad_files.append(os.path.join(root, file))
    
    file = h5ad_files[job_number]
    print(f"Reading file: {file}")
    adata = sc.read_h5ad(file)
    print(f"adata shape: {adata.shape}")
    adata.var['ensembl_id'] = adata.var['ensemblid']
    adata.var['ensembl_id'] = adata.var['ensembl_id'].str.split('.').str[0]

    model_dir     = Path("/nfs/turbo/umms-indikar/shared/projects/foundation_models/scGPT_human")
    cell_type_key = list(adata.obs.columns) #"dataset"
    gene_col      = "index"

    embedAdscGPT = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col=gene_col,
        obs_to_save=cell_type_key,  # optional arg, only for saving metainfo
        batch_size=64,
        return_new_adata=True,
    )
    embedAdscGPT.write(os.path.join(OUTPUTPATH, os.path.splitext(os.path.basename(file))[0] + '_scgpt.h5ad'))

    print('Job well done')
    sys.exit(0)


if __name__ == "__main__":
    main()
