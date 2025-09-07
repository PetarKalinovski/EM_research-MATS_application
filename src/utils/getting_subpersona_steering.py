import sys
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import pickle
import numpy as np
import torch


def get_cluster_steering(
    cluster_id: int, n_clusters: int, layer: int = 20
) -> torch.Tensor:
    """
    Generate a steering prompt for a given cluster ID.
    Args:
        cluster_id (int): The identifier for the cluster.
        n_clusters (int): The total number of clusters.
        layer (int, optional): The layer the activations are taken from. Defaults to 20.
    Returns:
        torch.Tensor: The steering vector for the specified cluster.
    """
    try:
        logger.info(
            f"Loading dataset to compute steering vector for cluster {cluster_id} at layer {layer}..."
        )
        project_root = Path(__file__).resolve().parent.parent.parent
        dataframe = pickle.load(
            open(
                project_root
                / "results"
                / "final_dataset_with_em_scores_and_clusters.pkl",
                "rb",
            )
        )
        logger.info("Dataset loaded successfully.")

        n_clusters = "cluster_" + str(n_clusters)
        cluster_data = dataframe[dataframe[n_clusters] == cluster_id]
        logger.info(
            f"Data filtered for cluster {cluster_id}. Number of samples: {len(cluster_data)}"
        )

        logger.info(f"Computing steering vector for layer {layer}...")
        em_activations_layer = "em_activations_layer_" + str(layer)
        cluster_em = np.vstack(cluster_data[em_activations_layer].to_list())
        cluster_base = np.vstack(
            cluster_data["base_activations_layer_" + str(layer)].to_list()
        )
        logger.info(
            f"Activation shapes - EM: {cluster_em.shape}, Base: {cluster_base.shape}"
        )

        steering_vector = np.mean(cluster_em, axis=0) - np.mean(cluster_base, axis=0)

        logger.info(
            f"Steering vector for cluster {cluster_id} at layer {layer} computed."
        )
        logger.info(f"Vector shape: {steering_vector.shape}")

        return torch.tensor(steering_vector, dtype=torch.bfloat16)

    except Exception as e:
        logger.error(f"Error in computing steering vector: {e}")
        raise
