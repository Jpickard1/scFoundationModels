import argparse
import scanpy as sc
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--n_clusters', type=int, required=True, help='Number of clusters')
parser.add_argument('--run', type=int, required=True, help='Run number')
args = parser.parse_args()

# Load your AnnData object
# Replace with your actual code to load the data
adata = sc.read('/nfs/turbo/umms-indikar/shared/projects/DARPA_AI/in-silico-reprogramming/unperturbed/fibroblast.h5ad')  # Adjust as necessary

# Normalize and log-transform the data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Extract the count matrix and standardize the features
X = adata.X
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=args.n_clusters, init='k-means++', n_init=1, random_state=42 + args.run)
cluster_labels = kmeans.fit_predict(X_scaled)
adata.obs[f'kmeans_{args.n_clusters}_{args.run}'] = cluster_labels

# Calculate silhouette score
output_file = "/home/oliven/scFoundationModels/notebooks/reprogramming/output_kmeans/silhouette_scores.txt"
score = silhouette_score(X_scaled, cluster_labels) if args.n_clusters > 1 else None
with open(output_file, "a") as f:  # Use "a" to append to the file
    f.write(f"Silhouette score for kmeans_{args.n_clusters}_{args.run}: {score:.4f}\n")

# Save the silhouette score in adata.uns
if 'silhouette_scores' not in adata.uns:
    adata.uns['silhouette_scores'] = {}
adata.uns['silhouette_scores'][f'kmeans_{args.n_clusters}_{args.run}'] = score

# Optionally save the updated adata object after each run
adata.write('/nfs/turbo/umms-indikar/shared/projects/DARPA_AI/in-silico-reprogramming/fibroblast_typing_9_20_n/output_kmeans/ts_fb_many_clusters.h5ad')
