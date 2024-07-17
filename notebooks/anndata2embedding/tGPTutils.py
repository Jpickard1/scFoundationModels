import re
import anndata as ad
import numpy as np
from torch.utils.data import DataLoader, Dataset

class LineDataset(Dataset):
    def __init__(self, lines):
        self.lines = lines
        self.regex = re.compile(r'\-|\.')
    def __getitem__(self, i):
        return self.regex.sub('_', self.lines[i])
    def __len__(self):
        return len(self.lines)

def get_top_genes_per_cell(adata, top_n=256):
    # Extract the data matrix
    X = adata.X
    
    # Initialize a list to hold the top genes for each cell
    top_genes_per_cell = []

    for i in range(X.shape[0]):
        # Get the expression values for the current cell
        cell_expression = X[i, :]
        
        # Get the indices of the top N expression values
        top_gene_indices = np.argsort(cell_expression)[-top_n:]
        
        # Get the gene names corresponding to the top indices
        top_gene_names = adata.var_names[top_gene_indices]
        
        top_genes_per_cell.append(" ".join(list(top_gene_names)))
    
    return top_genes_per_cell