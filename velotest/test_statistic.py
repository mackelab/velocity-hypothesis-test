from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


def cosine_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)

    similarities = np.einsum("...nd, ...md -> ...nm", x, y)
    similarities /= np.maximum(x_norm * y_norm, eps)

    return similarities


def cos_directionality_one_cell_batch_same_neighbors(
    expression: np.ndarray,
    velocity_vector: np.ndarray,
    expressions_neighbours: np.ndarray,
) -> np.ndarray:
    """
    Calculates the cosine similarity between the velocity of a cell and multiple sets of other cells
    (e.g., in the neighborhood of the cell). Every set is assumed to have same number of neighbors.

    :param expression: vector of gene expressions of the cell
    :param velocity_vector: velocity vector of the cell, not the position (x+v) of the velocity
    :param expressions_neighbours: (#neighborhoods, #neighbors, #genes)
    :return:
    """
    # return cosine_similarity(expressions_neighbours - expression, velocity_vector[..., None, :])
    number_neighborhoods = expressions_neighbours.shape[0]
    number_neighbors_per_neighborhood = expressions_neighbours.shape[1]

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    return cos(
        expressions_neighbours - expression,
        velocity_vector[None, None, :].expand(number_neighborhoods, number_neighbors_per_neighborhood, -1)
    )


def cos_directionality_one_cell_one_neighborhood(
    expression: torch.Tensor,
    velocity_vector: torch.Tensor,
    expressions_neighbours: torch.Tensor,
):
    """
    Calculates the cosine similarity between the velocity of a cell and one set of other cells
    (e.g., in the neighborhood of the cell).

    :param expression: vector of gene expressions of the cell
    :param velocity_vector: velocity vector of the cell, not the position (x+v) of the velocity
    :param expressions_neighbours: (#neighbors, #genes)
    :return:
    """
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    number_neighbors = expressions_neighbours.shape[0]
    return cos(expressions_neighbours - expression, velocity_vector[None, :].expand(number_neighbors, -1))


def mean_cos_directionality_varying_neighborhoods_same_neighbors(
    expression: torch.Tensor,
    velocity_vector: torch.Tensor,
    neighborhoods: dict,
    # original_indices_cells,
) -> dict[int, np.ndarray]:
    """
    Mean cos directionality for a varying number of neighbors in a neighborhood across cells but with same number
    of neighbors per cell:
    Calculates the mean cosine similarity between the velocity of a cell and
    multiple sets of other cells (e.g., in the neighborhood of the cell).
    Every set is assumed to have same number of neighbors.

    :param expression: expression of all cells
    :param velocity_vector: velocity vectors of all cells, not position (x+v) of the velocity
    :param neighborhoods: Neighborhoods of selected cells. list of length #cells with (#neighborhoods, #neighbors).
    :param original_indices_cells: indices of the selected cells in the original expression matrix
    :return:
    """
    # number_cells = len(neighborhoods)
    # number_neighborhoods = neighborhoods[0].shape[0]
    # mean_cos_neighborhoods = np.zeros((number_cells, number_neighborhoods))
    mean_cos_neighborhoods = {}

    for cell_idx in tqdm(neighborhoods):
        cosine_similarities = cos_directionality_one_cell_batch_same_neighbors(
            expression[cell_idx],
            velocity_vector[cell_idx],
            expression[neighborhoods[cell_idx]],
        )
        # Since we are comparing to a single velocity vector, the final shape
        # will be (B, N, 1), so we can get rid of the last dimension
        # cosine_similarities = cosine_similarities.squeeze(-1)
        # Compute mean cosine simility in all negihborhoods
        mean_cos_neighborhoods[cell_idx] = torch.mean(cosine_similarities, dim=-1).numpy()

    return mean_cos_neighborhoods



def mean_cos_directionality_varying_neighbors(
    expression: torch.Tensor,
    velocity_vector: torch.Tensor,
    neighborhoods: dict,
    cosine_empty_neighborhood: Optional[float] = 2,
):
    """
    Mean cos directionality for a varying number of neighbors in the neighborhoods across cells:
    Calculates the cosine similarity between the velocity of a cell and multiple sets of other cells
    (e.g., in the neighborhood of the cell). Every set can have a different number of cells.

    :param expression: expression of all cells
    :param velocity_vector: velocity vectors of all cells, not position (x+v) of the velocity
    :param neighborhoods: Neighborhoods of selected cells. list of length #cells with lists of varying #neighbors.
    :return:
    """
    # number_cells = len(neighborhoods)
    # number_neighborhoods = len(neighborhoods[next(iter(neighborhoods))])
    #
    # if cosine_empty_neighborhood is not None:
    #     mean_cos_neighborhoods = torch.zeros((number_cells, number_neighborhoods))
    # else:
    #     mean_cos_neighborhoods = []

    mean_cos_neighborhoods = {}

    for cell_idx in tqdm(neighborhoods):
        cos_similarities = np.full(len(neighborhoods[cell_idx]), fill_value=np.nan)
        for neighborhood_id, neighborhood in enumerate(neighborhoods[cell_idx]):
            cos_similarities[neighborhood_id] = torch.mean(
                cos_directionality_one_cell_one_neighborhood(
                    expression[cell_idx], velocity_vector[cell_idx], expression[neighborhood]
                ))
        mean_cos_neighborhoods[cell_idx] = cos_similarities

    return mean_cos_neighborhoods
