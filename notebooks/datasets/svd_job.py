import sys
import numpy as np
import pickle
from scipy.sparse import issparse
import anndata as ad

def compute_svd(file, category, model):
    result = {
        'file': file,
        'data': category,
        'model': model,
        'numCells': None,
        'dimension': None,
        'sigmas': None
    }

    try:
        adata = ad.read_h5ad(file)
        result['numCells'] = adata.X.shape[0]
        result['dimension'] = adata.X.shape[1]
        X = adata.X
        if issparse(X):
            X = X.toarray()

        sigmas = np.linalg.svd(X, compute_uv=False)
        result['sigmas'] = sigmas
    except Exception as e:
        print(f"Error processing {file}: {e}")

    return result

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python svd_job.py <file> <category> <model>")
        sys.exit(1)

    file = sys.argv[1]
    category = sys.argv[2]
    model = sys.argv[3]

    result = compute_svd(file, category, model)

    # Save result to a pickle file
    output_file = f'/nfs/turbo/umms-indikar/shared/projects/foundation_models/experiments/tabulaSapiens/results/result_{category}_{model}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"SVD computation completed for {file}. Results saved to {output_file}")
