import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA


def load_clustering_results(results_dir="results"):
    """Load all clustering results from pickle files"""
    results_dir = Path(results_dir)
    cluster_counts = [3, 4, 5, 7, 15, 50]

    clustering_data = {}

    for n_clusters in cluster_counts:
        pickle_file = results_dir / f"em_pca_clusters_{n_clusters}.pkl"

        if pickle_file.exists():
            print(f"Loading {pickle_file}")
            with open(pickle_file, "rb") as f:
                data = pickle.load(f)
                clustering_data[n_clusters] = data
        else:
            print(f"Warning: {pickle_file} not found")

    return clustering_data


def apply_second_pca_reduction(clustering_data, n_components=2):
    """Apply a second PCA reduction to the already PCA-reduced data"""

    reduced_data = {}

    for n_clusters, data in clustering_data.items():
        print(f"Applying second PCA to {n_clusters}-cluster data...")

        # Apply second PCA to the 50-component data
        pca_2nd = PCA(n_components=n_components)
        em_2d = pca_2nd.fit_transform(data['em_reduced'])

        print(f"  Second PCA explained variance: {pca_2nd.explained_variance_ratio_}")
        print(f"  Total variance explained: {pca_2nd.explained_variance_ratio_.sum():.3f}")

        # Store the results
        reduced_data[n_clusters] = {
            'em_2d': em_2d,
            'clusters': data['clusters'],
            'cluster_centers': data['cluster_centers'],
            'pca_2nd': pca_2nd,
            'original_em_reduced': data['em_reduced']
        }

        # Also transform cluster centers if they exist
        if data['cluster_centers'] is not None:
            centers_2d = pca_2nd.transform(data['cluster_centers'])
            reduced_data[n_clusters]['centers_2d'] = centers_2d

    return reduced_data


def create_all_visualizations(clustering_data, output_dir="cluster_plots"):
    """Create all types of visualizations"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Apply second PCA reduction
    reduced_data = apply_second_pca_reduction(clustering_data)

    # 1. Create overview plot (all clusters in one figure)
    create_overview_plot(reduced_data, output_dir)

    # 2. Create individual plots for each cluster count
    create_individual_plots(reduced_data, output_dir)

    # 3. Create detailed analysis plots
    create_detailed_analysis_plots(reduced_data, output_dir)

    # 4. Create comparison plot (original PC1-PC2 vs new 2D)
    create_comparison_plots(clustering_data, reduced_data, output_dir)

    return reduced_data


def get_distinct_colors(n_colors):
    """Get highly distinct colors for better cluster visualization"""
    if n_colors <= 10:
        # Hand-picked distinct colors for small cluster counts
        distinct_colors = [
            '#FF0000',  # Bright Red
            '#0080FF',  # Bright Blue
            '#00CC00',  # Bright Green
            '#FF8000',  # Bright Orange
            '#8000FF',  # Purple
            '#FF0080',  # Hot Pink
            '#00FFFF',  # Cyan
            '#FFFF00',  # Yellow
            '#FF8080',  # Light Red
            '#80FF80'  # Light Green
        ]
        return distinct_colors[:n_colors]
    else:
        # Use tab20 for larger cluster counts (more distinct than Set3)
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_colors, 20)))
        if n_colors > 20:
            # Add more colors from other colormaps
            extra_colors = plt.cm.tab20b(np.linspace(0, 1, n_colors - 20))
            colors = np.vstack([colors, extra_colors])
        return colors


def create_overview_plot(reduced_data, output_dir, figsize=(20, 12)):
    """Create overview plot with all clustering results"""

    n_plots = len(reduced_data)
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    axes_flat = axes.flatten()

    plot_idx = 0

    for n_clusters in sorted(reduced_data.keys()):
        data = reduced_data[n_clusters]

        pc1 = data['em_2d'][:, 0]
        pc2 = data['em_2d'][:, 1]
        clusters = data['clusters']

        ax = axes_flat[plot_idx]

        # Get distinct colors for this number of clusters
        unique_clusters = np.unique(clusters)
        colors = get_distinct_colors(len(unique_clusters))

        # Create scatter plot with cluster colors
        for i, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            ax.scatter(pc1[mask], pc2[mask],
                       c=colors[i],
                       alpha=0.8,
                       s=60,  # Increased from 25
                       label=f'Cluster {cluster_id}',
                       edgecolors='black',  # Changed from white to black for better contrast
                       linewidth=0.5)

        # Plot cluster centers if available
        if 'centers_2d' in data:
            centers = data['centers_2d']
            ax.scatter(centers[:, 0], centers[:, 1],
                       c='black',  # Changed to black for better visibility
                       marker='X',
                       s=200,  # Increased from 150
                       edgecolors='white',
                       linewidth=2,
                       label='Centers',
                       zorder=5)

        ax.set_xlabel('PC1 (2nd PCA)')
        ax.set_ylabel('PC2 (2nd PCA)')
        ax.set_title(f'{n_clusters} Clusters\n{len(clusters)} points')
        ax.grid(True, alpha=0.3)

        # Add legend only for smaller cluster counts
        if n_clusters <= 7:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle('EM Activation Clusters - PCA Projection', fontsize=16, y=0.98)
    plt.tight_layout()

    # Save plot
    output_file = output_dir / "overview_2nd_pca.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved overview plot: {output_file}")
    plt.show()


def create_individual_plots(reduced_data, output_dir):
    """Create individual plots for each cluster count"""

    for n_clusters in sorted(reduced_data.keys()):
        data = reduced_data[n_clusters]

        plt.figure(figsize=(12, 9))

        pc1 = data['em_2d'][:, 0]
        pc2 = data['em_2d'][:, 1]
        clusters = data['clusters']

        # Get distinct colors for this number of clusters
        unique_clusters = np.unique(clusters)
        colors = get_distinct_colors(len(unique_clusters))

        # Create scatter plot
        for i, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            plt.scatter(pc1[mask], pc2[mask],
                        c=colors[i],
                        alpha=0.8,
                        s=80,  # Increased from 40
                        label=f'Cluster {cluster_id}',
                        edgecolors='black',  # Changed from white
                        linewidth=0.7)  # Increased thickness

        # Plot centers
        if 'centers_2d' in data:
            centers = data['centers_2d']
            plt.scatter(centers[:, 0], centers[:, 1],
                        c='black', marker='X', s=250,  # Increased from 200
                        edgecolors='white', linewidth=2.5,  # Increased thickness
                        label='Centers', zorder=5)

        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        plt.title(f'EM Activations: {n_clusters} Clusters (Layer 20)\n'
                  f'{len(clusters)} data points - PCA Projection', fontsize=14)

        if n_clusters <= 15:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_file = output_dir / f"clusters_{n_clusters}_2nd_pca.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved individual plot: {output_file}")
        plt.show()


def create_detailed_analysis_plots(reduced_data, output_dir):
    """Create detailed analysis plots for each clustering result"""

    for n_clusters in sorted(reduced_data.keys()):
        data = reduced_data[n_clusters]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        pc1 = data['em_2d'][:, 0]
        pc2 = data['em_2d'][:, 1]
        clusters = data['clusters']

        # Get distinct colors for this number of clusters
        unique_clusters = np.unique(clusters)
        colors = get_distinct_colors(len(unique_clusters))

        # Main scatter plot
        for i, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            ax1.scatter(pc1[mask], pc2[mask],
                        c=colors[i],
                        alpha=0.8,
                        s=70,  # Increased from 50
                        label=f'Cluster {cluster_id}',
                        edgecolors='black',  # Changed from white
                        linewidth=0.6)

        # Plot centers
        if 'centers_2d' in data:
            centers = data['centers_2d']
            ax1.scatter(centers[:, 0], centers[:, 1],
                        c='black', marker='X', s=250,  # Increased from 200
                        edgecolors='white', linewidth=2.5,
                        label='Centers', zorder=5)

        ax1.set_xlabel('PC1 ')
        ax1.set_ylabel('PC2)')
        ax1.set_title(f'{n_clusters} Clusters -  PCA Projection')
        if n_clusters <= 7:
            ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cluster size distribution
        cluster_sizes = np.bincount(clusters)
        bars = ax2.bar(range(len(cluster_sizes)), cluster_sizes,
                       color=colors[:len(cluster_sizes)],  # Use our distinct colors
                       alpha=0.8, edgecolor='black', linewidth=0.8)  # Increased edge thickness
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Points')
        ax2.set_title('Cluster Size Distribution')
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, size in zip(bars, cluster_sizes):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(size), ha='center', va='bottom', fontweight='bold')

        # PC1 distribution by cluster
        for i, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            ax3.hist(pc1[mask], alpha=0.7, bins=20,
                     color=colors[i], label=f'Cluster {cluster_id}',
                     edgecolor='black', linewidth=0.8)  # Increased edge thickness
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('Count')
        ax3.set_title('PC1 Distribution by Cluster')
        if n_clusters <= 7:
            ax3.legend()
        ax3.grid(True, alpha=0.3)

        # PC2 distribution by cluster
        for i, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            ax4.hist(pc2[mask], alpha=0.7, bins=20,
                     color=colors[i], label=f'Cluster {cluster_id}',
                     edgecolor='black', linewidth=0.8)  # Increased edge thickness
        ax4.set_xlabel('PC2')
        ax4.set_ylabel('Count')
        ax4.set_title('PC2 Distribution by Cluster')
        if n_clusters <= 7:
            ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Detailed Analysis: {n_clusters} Clusters',
                     fontsize=16, y=0.98)
        plt.tight_layout()

        # Save plot
        output_file = output_dir / f"detailed_analysis_{n_clusters}_2nd_pca.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved detailed analysis: {output_file}")
        plt.show()


def print_cluster_statistics(clustering_data):
    """Print detailed statistics for each clustering result"""

    print("=" * 80)
    print("CLUSTER ANALYSIS STATISTICS")
    print("=" * 80)

    for n_clusters in sorted(clustering_data.keys()):
        data = clustering_data[n_clusters]
        clusters = data['clusters']

        print(f"\n{n_clusters} Clusters:")
        print("-" * 30)

        # Basic stats
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        print(f"Total points: {len(clusters)}")
        print(f"Actual clusters found: {len(unique_clusters)}")

        # Cluster size statistics
        cluster_dict = dict(zip(unique_clusters, counts))
        print(f"Cluster sizes: {cluster_dict}")
        print(f"Largest cluster: {np.max(counts)} points ({np.max(counts) / len(clusters) * 100:.1f}%)")
        print(f"Smallest cluster: {np.min(counts)} points ({np.min(counts) / len(clusters) * 100:.1f}%)")
        print(f"Average cluster size: {np.mean(counts):.1f}")
        print(f"Std dev of cluster sizes: {np.std(counts):.1f}")

        # Balance metric (coefficient of variation)
        cv = np.std(counts) / np.mean(counts)
        print(f"Cluster balance (lower = more balanced): {cv:.3f}")

        # PCA info
        if 'pca' in data:
            pca = data['pca']
            print(f"Original PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")


def main():
    # Load all clustering results
    clustering_data = load_clustering_results()

    if not clustering_data:
        print("No clustering data found! Make sure your pickle files are in the 'results' directory.")
        return

    print(f"Loaded clustering results for: {list(clustering_data.keys())} clusters")

    # Print statistics
    print_cluster_statistics(clustering_data)

    # Create all visualizations
    print("\nCreating all visualizations with second PCA projection...")
    reduced_data = create_all_visualizations(clustering_data)

    print("\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    print("Generated the following types of plots:")
    print("1. Overview plot showing all cluster counts")
    print("2. Individual plots for each cluster count")
    print("3. Detailed analysis plots (4-panel layout)")
    print("4. Comparison plots (original vs second PCA)")
    print("\nAll plots saved in 'cluster_plots/' directory")
    print("Second PCA projections provide better visualization of cluster separation!")


if __name__ == "__main__":
    main()