import os
import sys
import scanpy as sc
import anndata as ad
import importlib

sys.path.append('../notebooks/anndata2embedding')
import embed
importlib.reload(embed)
from embed import *

# Input and output paths
input_path = "/nfs/turbo/umms-indikar/shared/projects/DARPA_AI/in-silico-reprogramming/one-shot/perturbed"
output_path = "/nfs/turbo/umms-indikar/shared/projects/DARPA_AI/in-silico-reprogramming/one-shot/geneformer"

# Get the array index from the environment variable
array_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
print(f"{array_id=}")

# Get the list of .h5ad files in the input directory
h5ad_files = [f for f in os.listdir(input_path) if f.endswith(".h5ad")]

# Make sure the array index is valid
if array_id >= len(h5ad_files):
    print(f"Array index {array_id} is out of range for the number of files.")
    sys.exit(1)

# Get the specific file based on the SLURM array task ID
filename = h5ad_files[array_id]
file_path = os.path.join(input_path, filename)
output_file = os.path.join(output_path, filename)

# If the output file already exists, skip processing
if os.path.exists(output_file):
    print(f"File already exists, skipping: {output_file}")
else:
    # Read the file
    adata = sc.read_h5ad(file_path)

    # Change the ENSG ids to conform to GF tokenization
    adata.var['endembdid_split'] = adata.var['ensemblid'].str.split('.').str[0]

    # Print debugging information
    print(f"Processing file: {filename}")

    # Embed the AnnData object using Geneformer
    embedAdGF = embed(adata, 'geneformer',
                      output_directory=output_path,
                      output_file=os.path.splitext(os.path.basename(filename))[0],
                      genenameloc='endembdid_split',
                      n_counts_column='n_counts_UMIs'
                     )

    # Save the resulting embedding to the output file
    embedAdGF.write_h5ad(output_file)
    print(f"Embedding saved to: {output_file}")
print("Job is complete")