import numpy as np
import torch

from velotest.neighbors import find_neighbors, find_neighbors_in_direction_of_velocity
from velotest.test_statistic import mean_cos_directionality_varying_neighborhoods


#: ArrayLike[NDArray]
def p_values(test_statistics_velocity, test_statistics_random):
    """
    Compute p-values using the test statistics from the permutations.
    @param test_statistics_velocity: (#cells)
    @param test_statistics_random: (#cells, #neighborhoods)
    @return:
    """
    return torch.sum(test_statistics_random >= test_statistics_velocity.unsqueeze(1), axis=-1) / \
        test_statistics_random.shape[-1]


def benjamini_hochberg(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction for multiple testing.
    @param p_values: (#cells)
    @param alpha: significance level
    @return:
    """
    sorted_p_values = np.sort(p_values)
    m = len(p_values)
    h0_rejected = np.zeros(m, dtype=bool)
    for i in range(m):
        if sorted_p_values[i] <= (i + 1) / m * alpha:
            h0_rejected[i] = True
        else:
            break
    return h0_rejected[p_values.argsort()]


#:ArrayLike[int]
def run_hypothesis_test(X_expr, X_velo_vector, Z_expr, Z_velo_position,
                        number_neighborhoods=100, number_neighbors_to_sample_from=50,
                        correction='benjamini–hochberg', seed=0):
    """
    Samples random neighborhoods for every cell and uses the high-dimensional cosine similarity between
    the velocity of each cell and the cells in the direction of the velocity (in 2D) as test statistic.

    @param X_expr: high-dimensional expressions
    @param X_velo_vector: high-dimensional velocity vector, not position (x+v)
    @param Z_expr: embedding for expressions
    @param number_neighborhoods: number of neighborhoods used to define null distribution
    @param number_neighbors_to_sample_from: number of neighbors to sample neighborhoods from and
    to look for neighbors in direction of velocity
    @param batch_size: batch size for computing cosine similarity
    @param correction: correction method for multiple testing. 'benjamini–hochberg' or None
    @return: p_values_ (p-values from test (not corrected), cells where test couldn't be run are assigned a value of 2),
    h0_rejected , test_statistics_velocity, test_statistics_random, neighborhoods
    """
    number_cells = X_expr.shape[0]

    nn_indices = find_neighbors(Z_expr, k_neighbors=number_neighbors_to_sample_from)
    neighbors_in_direction_of_velocity = find_neighbors_in_direction_of_velocity(Z_expr, Z_velo_position, nn_indices)

    non_empty_neighborhoods_bool = [len(neighborhood) != 0 for neighborhood in neighbors_in_direction_of_velocity]
    non_empty_neighborhoods_bool = np.array(non_empty_neighborhoods_bool)
    non_empty_neighborhoods_indices = np.where(non_empty_neighborhoods_bool)[0]
    neighbors_in_direction_of_velocity = [neighbors_in_direction_of_velocity[index] for index in
                                          non_empty_neighborhoods_indices]

    np.random.seed(seed)

    neighborhoods_random = []  # #cells long list of (#neighborhoods, #neighbors_per_neighborhood)
    for cell, neighborhood in zip(non_empty_neighborhoods_indices, neighbors_in_direction_of_velocity):
        number_neighbors_per_neighborhood = len(neighborhood)
        neighborhoods_random.append(
            np.random.choice(nn_indices[cell], size=(number_neighborhoods, number_neighbors_per_neighborhood)))

    neighborhoods = [np.concatenate([np.expand_dims(in_direction_of_velocity, axis=0), random], axis=0) for
                     in_direction_of_velocity, random in
                     zip(neighbors_in_direction_of_velocity, neighborhoods_random)]

    test_statistics = mean_cos_directionality_varying_neighborhoods(torch.tensor(X_expr), torch.tensor(X_velo_vector),
                                                                    neighborhoods, non_empty_neighborhoods_indices)
    test_statistics_velocity = test_statistics[:, 0]
    test_statistics_random = test_statistics[:, 1:]
    p_values_ = p_values(test_statistics_velocity, test_statistics_random).numpy()
    if correction == 'benjamini–hochberg':
        h0_rejected = benjamini_hochberg(p_values_)
    elif correction is None:
        h0_rejected = None
    else:
        raise ValueError(f"Unknown correction method: {correction}. Use 'benjamini–hochberg' or None.")

    p_values_all = 2 * np.ones(number_cells)
    p_values_all[non_empty_neighborhoods_bool] = p_values_
    if h0_rejected is not None:
        h0_rejected_all = np.zeros(number_cells, dtype=bool)
        h0_rejected_all[non_empty_neighborhoods_bool] = h0_rejected
    else:
        h0_rejected_all = None
    return p_values_all, h0_rejected_all, test_statistics_velocity.numpy(), test_statistics_random.numpy(), neighborhoods


def run_hypothesis_test_on(adata, ekey='Ms', vkey='velocity', basis='umap', **kwargs):
    """
    Runs the hypothesis test using high dimensional expressions, high dimensional velocity,
    and the embeddings from an adata object. For details, see `run_hypothesis_test`.

    :param adata: Anndata object containing high dimensional data and embeddings.
    :param ekey: Name of layer in adata object containing high dimensional expression data.
    :param vkey: Name of layer in adata object containing high dimensional velocity data.
    :param basis: Name of embedding.
    :param kwargs: Additional arguments for `run_hypothesis_test`.
    :return: See `run_hypothesis_test`.
    """
    X_expr = adata.layers[ekey]
    X_velo = adata.layers[vkey]
    Z_expr = adata.obsm[basis]
    Z_velo_position = adata.obsm[basis] + adata.obsm[f'{basis}_velocity']

    X_expr = torch.tensor(X_expr)
    X_velo_vector = torch.tensor(X_velo_vector)
    Z_expr = torch.tensor(Z_expr)
    Z_velo_position = torch.tensor(Z_velo_position)

    return run_hypothesis_test(X_expr, X_velo_vector, Z_expr, Z_velo_position, **kwargs)
