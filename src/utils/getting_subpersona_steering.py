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


def get_subset_steering(
        subset_name: str, layer: int = 20) -> torch.Tensor:
    """
    Generate a steering prompt for a given subset name.
    Args:
        subset_name (str): The identifier for the subset.
        layer (int, optional): The layer the activations are taken from. Defaults to 20.
    Returns:
        torch.Tensor: The steering vector for the specified subset.
    """
    try:
        logger.info(
            f"Loading dataset to compute steering vector for subset {subset_name} at layer {layer}..."
        )
        project_root = Path(__file__).resolve().parent.parent.parent
        dataframe = pickle.load(
            open(
                project_root
                / "results"
                / "persona_data_14b_with_data_subsets_column.pkl",
                "rb",
            )
        )
        logger.info("Dataset loaded successfully.")

        subset_data = dataframe[dataframe["subset"] == subset_name]
        logger.info(
            f"Data filtered for subset {subset_name}. Number of samples: {len(subset_data)}"
        )

        logger.info(f"Computing steering vector for layer {layer}...")
        em_activations_layer = "em_activations_layer_" + str(layer)
        subset_em = np.vstack(subset_data[em_activations_layer].to_list())
        subset_base = np.vstack(
            subset_data["base_activations_layer_" + str(layer)].to_list()
        )
        logger.info(
            f"Activation shapes - EM: {subset_em.shape}, Base: {subset_base.shape}"
        )

        steering_vector = np.mean(subset_em, axis=0) - np.mean(subset_base, axis=0)

        logger.info(
            f"Steering vector for subset {subset_name} at layer {layer} computed."
        )
        logger.info(f"Vector shape: {steering_vector.shape}")

        return torch.tensor(steering_vector, dtype=torch.bfloat16)

    except Exception as e:
        logger.error(f"Error in computing steering vector: {e}")
        raise


