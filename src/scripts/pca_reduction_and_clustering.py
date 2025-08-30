import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import pickle

def analyze_activation_clusters(df, layer=20, n_components=50, n_clusters=5):
    """
    Find natural clusters in EM model activations

    Args:
        df (pd.DataFrame): DataFrame containing EM activations
        layer (int): Layer number to analyze
        n_components (int): Number of PCA components
        n_clusters (int): Number of clusters for KMeans
    Returns:
        dict: Dictionary with PCA model, reduced activations, cluster labels, and cluster centers
    """
    # Get EM activations for the target layer
    em_activations = np.vstack(df[f'em_activations_layer_{layer}'])
    print(f"EM activations shape: {em_activations.shape}")

    # PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    em_reduced = pca.fit_transform(em_activations)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(em_reduced)

    return {
        'pca': pca,
        'em_reduced': em_reduced,
        'clusters': clusters,
        'cluster_centers': kmeans.cluster_centers_
    }


def main():
    df = pd.read_pickle("results/final_dataset_with_em_scores.pkl")
    print(f"Loaded dataset with {len(df)} entries")

    if f'em_activations_layer_20' not in df.columns:
        raise ValueError("EM activations for layer 20 not found in DataFrame")

    for n_clusters in [3, 4, 5,7,15,50]:
        print(f"\n=== ANALYZING WITH {n_clusters} CLUSTERS ===")
        results = analyze_activation_clusters(df, layer=20, n_clusters=n_clusters)

        df[f'cluster_{n_clusters}'] = results['clusters']

        unique, counts = np.unique(results['clusters'], return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        print(f"Cluster sizes: {cluster_sizes}")

        print(f"Cluster centers (first 2 components):\n{results['cluster_centers'][:, :2]}")


        output_path = Path(f"results/em_pca_clusters_{n_clusters}.pkl").absolute()
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Saved PCA and clustering results to {output_path}")


    pickle.dump(df, open(Path("results/final_dataset_with_em_scores_and_clusters.pkl").absolute(), "wb"))

if __name__ == "__main__":
    main()