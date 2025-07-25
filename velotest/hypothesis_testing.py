from typing import Optional

import numpy as np
import torch
from scipy.stats import false_discovery_control

from velotest.neighbors import find_neighbors, find_neighbors_in_direction_of_velocity, \
    find_neighbors_in_direction_of_velocity_multiple
from velotest.test_statistic import mean_cos_directionality_varying_neighborhoods_same_neighbors, \
    mean_cos_directionality_varying_neighbors, mean_cos_directionality_varying_neighbors_parallel


#: ArrayLike[NDArray]
def p_values(test_statistics_velocity, test_statistics_random):
    """
    Compute p-values using the test statistics from the permutations.

    :param test_statistics_velocity: (#cells)
    :param test_statistics_random: (#cells, #neighborhoods)
    :return:
    """
    return torch.sum(test_statistics_random >= test_statistics_velocity.unsqueeze(1), axis=-1) / \
        test_statistics_random.shape[-1]


def p_values_list(test_statistics_velocity, test_statistics_random):
    return torch.tensor(
        [torch.sum(torch.tensor(set_of_random_statistics) >= observed_statistic) / len(set_of_random_statistics) for
         observed_statistic, set_of_random_statistics in zip(test_statistics_velocity, test_statistics_random)])


def correct_for_multiple_testing(pvals, correction):
    if correction is None:
        return pvals

    elif correction == 'benjamini–hochberg':
        return false_discovery_control(pvals)

    elif correction == 'bonferroni':
        n_tests = len(pvals)
        pvals_corrected = pvals * n_tests
        np.clip(pvals_corrected, a_min=None, a_max=1, out=pvals_corrected)
        return pvals_corrected

    else:
        raise ValueError(
            f"Unknown correction method: '{correction}'. Supported methods "
            f"include 'benjamini–hochberg', 'bonferroni', and None."
        )


#:ArrayLike[int]
def run_hypothesis_test(
    X_expr,
    X_velo_vector,
    Z_expr,
    Z_velo_position,
    number_neighborhoods=1000,
    number_neighbors_to_sample_from=50,
    threshold_degree=22.5,
    exclusion_degree: Optional[float] = None,
    null_distribution='neighbors',
    correction='benjamini–hochberg',
    alpha=0.05,
    cosine_empty_neighborhood=2,
    seed=0,
):
    """
    Samples random neighborhoods for every cell and uses the high-dimensional cosine similarity between
    the velocity of each cell and the cells in the direction of the velocity (in 2D) as test statistic.

    :param X_expr: high-dimensional expressions
    :param X_velo_vector: high-dimensional velocity vector, not position (x+v)
    :param Z_expr: embedding for expressions
    :param Z_velo_position: embedding for velocity position (x+v)
    :param number_neighborhoods: number of neighborhoods used to define null distribution
    :param number_neighbors_to_sample_from: number of neighbors to sample neighborhoods from and
        to look for neighbors in direction of velocity
    :param threshold_degree: angle in degrees to define the cone around the velocity vector
        (angle of cone is 2*threshold_degree),
    :param exclusion_degree: angle in degrees to exclude random velocities which are too similar to
        the visualized velocity. 'None' uses all random velocities.
    :param null_distribution: 'neighbors' or 'velocities'. If 'neighbors', the neighborhoods are uniformly sampled from the neighbors.
        If 'velocities', random velocities are sampled and then the neighborhoods are defined by the neighbors in this direction.
    :param correction: correction method for multiple testing. 'benjamini–hochberg', 'bonferroni' or None
    :param alpha: significance level used for Benjamini-Hochberg or Bonferroni correction.
    :param cosine_empty_neighborhood: See `mean_cos_directionality_varying_neighbors`.
    :param seed: Random seed for reproducibility.
    :return:
        - ``p_values_`` (p-values from test (not corrected), cells where test couldn't be run are assigned a value of 2),
        - ``h0_rejected``
        - ``test_statistics_velocity``
        - ``test_statistics_random``
        - ``neighborhoods``
    """
    assert not (null_distribution == 'neighbors' and cosine_empty_neighborhood is None)

    if not isinstance(X_expr, torch.Tensor):
        X_expr = torch.tensor(X_expr)
    if not isinstance(X_velo_vector, torch.Tensor):
        X_velo_vector = torch.tensor(X_velo_vector)
    if not isinstance(Z_expr, torch.Tensor):
        Z_expr = torch.tensor(Z_expr)
    if not isinstance(Z_velo_position, torch.Tensor):
        Z_velo_position = torch.tensor(Z_velo_position)

    assert not (torch.isnan(X_expr).any()), ("X_expr contains NaN values. "
                                             "Please remove them before running the test.")
    assert not (torch.isnan(X_velo_vector).any()), ("X_velo_vector contains NaN values. "
                                                    "Please remove them before running the test.")
    assert not (torch.isnan(Z_expr).any()), ("Z_expr contains NaN values. "
                                             "Please remove them before running the test.")
    assert not (torch.isnan(Z_velo_position).any()), ("Z_velo_position contains NaN values. "
                                                      "Please remove them before running the test.")

    number_cells = X_expr.shape[0]

    nn_indices = find_neighbors(Z_expr, k_neighbors=number_neighbors_to_sample_from)
    neighbors_in_direction_of_velocity = find_neighbors_in_direction_of_velocity(Z_expr, Z_velo_position, nn_indices,
                                                                                 threshold_degree)

    non_empty_neighborhoods_bool = [len(neighborhood) != 0 for neighborhood in neighbors_in_direction_of_velocity]
    non_empty_neighborhoods_bool = np.array(non_empty_neighborhoods_bool)
    non_empty_neighborhoods_indices = np.where(non_empty_neighborhoods_bool)[0]
    neighbors_in_direction_of_velocity = [neighbors_in_direction_of_velocity[index] for index in
                                          non_empty_neighborhoods_indices]

    np.random.seed(seed)

    debug_dict = {}
    number_neighbors_per_velocity_neighborhood = [len(neighborhood) for neighborhood in
                                                  neighbors_in_direction_of_velocity]
    if null_distribution == 'neighbors':
        neighborhoods_random = []  # #cells long list of (#neighborhoods, #neighbors_per_neighborhood)
        for cell, number_neighbors in zip(non_empty_neighborhoods_indices, number_neighbors_per_velocity_neighborhood):
            neighborhoods_random.append(
                np.random.choice(nn_indices[cell], size=(number_neighborhoods, number_neighbors)))

        neighborhoods = [np.concatenate([np.expand_dims(in_direction_of_velocity, axis=0), random], axis=0) for
                         in_direction_of_velocity, random in
                         zip(neighbors_in_direction_of_velocity, neighborhoods_random)]
        test_statistics = mean_cos_directionality_varying_neighborhoods_same_neighbors(X_expr,
                                                                                       X_velo_vector,
                                                                                       neighborhoods,
                                                                                       non_empty_neighborhoods_indices)
    elif null_distribution == 'velocities':
        # Sample number_neighborhoods random velocities on unit circle for each cell and add them to Z_expr
        uniform = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([2.0]))
        angle = torch.pi * uniform.sample(sample_shape=(number_neighborhoods, number_cells,))
        x = torch.cos(angle)
        y = torch.sin(angle)
        Z_velo_position_random = Z_expr + torch.concatenate((x, y), dim=-1)
        Z_velo_position_random_transposed = Z_velo_position_random.detach().clone()
        Z_velo_position_random_transposed = torch.transpose(Z_velo_position_random_transposed, 1, 0)
        debug_dict['Z_velo_position_random'] = Z_velo_position_random_transposed.numpy()

        if exclusion_degree is not None:
            Z_velo_normalized = (Z_velo_position - Z_expr) / (Z_velo_position - Z_expr).norm(dim=1, keepdim=True)
            theta = torch.atan2(Z_velo_normalized[:, 1], Z_velo_normalized[:, 0])
            theta[theta < 0] += 2 * torch.pi

            mask_not_excluded = torch.logical_or(angle.squeeze() < theta - np.deg2rad(exclusion_degree),
                                                 angle.squeeze() > theta + np.deg2rad(exclusion_degree))
            mask_not_excluded = mask_not_excluded.T
            debug_dict['mask_not_excluded-all'] = mask_not_excluded

        neighborhoods_random_velocities = find_neighbors_in_direction_of_velocity_multiple(Z_expr,
                                                                                           Z_velo_position_random,
                                                                                           nn_indices,
                                                                                           threshold_degree)

        # Remove cells with empty neighborhood for visualized velocity
        neighborhoods_random_velocities = [neighborhoods_random_velocities[cell] for cell in
                                           non_empty_neighborhoods_indices]
        if exclusion_degree is not None:
            mask_not_excluded = mask_not_excluded[non_empty_neighborhoods_indices]

        # Merge neighborhoods (from velocity and random)
        # (list, list, Tensor)
        neighborhoods = []
        for neighbors_in_direction_of_velocity_cell, random_neighborhoods_cell in zip(
                neighbors_in_direction_of_velocity, neighborhoods_random_velocities):
            merged_neighborhoods_cell = [torch.tensor(neighbors_in_direction_of_velocity_cell)]
            merged_neighborhoods_cell.extend(random_neighborhoods_cell)
            neighborhoods.append(merged_neighborhoods_cell)
        if cosine_empty_neighborhood is not None:
            print("Cannot use parallelization with cosine_empty_neighborhood != None.")
            test_statistics, used_neighborhoods = mean_cos_directionality_varying_neighbors(
                X_expr,
                X_velo_vector,
                neighborhoods,
                non_empty_neighborhoods_indices,
                cosine_empty_neighborhood)
        else:
            test_statistics, used_neighborhoods = mean_cos_directionality_varying_neighbors_parallel(
                X_expr,
                X_velo_vector,
                neighborhoods,
                non_empty_neighborhoods_indices)

        if isinstance(test_statistics, torch.Tensor):
            test_statistics_debug_dict = test_statistics.detach().clone()
        else:
            test_statistics_debug_dict = test_statistics.copy()
        debug_dict['test_statistic_all'] = test_statistics_debug_dict
        debug_dict['used_neighborhoods'] = used_neighborhoods

        if exclusion_degree is not None:
            # Select test_statistics based on mask_not_excluded
            # We're first computing the test statistics for all neighborhoods, then selecting the ones
            # that are not excluded. If one only wants to run the test, this wastes some time, but it makes it easier
            # to compute the "optimal velocity" later and some of our visualisations.
            # First element is the test statistic for the visualized velocity, the rest are random neighborhoods.

            # Exclude the first neighborhood (the one in direction of the velocity) from the mask
            used_neighborhoods = used_neighborhoods[:, 1:]
            test_statistics = [
                [test_statistics_one_cell[i] for i in np.append([0], np.where(mask_one_cell[used_neighborhoods_one_cell])[0] + 1)] for
                mask_one_cell, test_statistics_one_cell, used_neighborhoods_one_cell in zip(mask_not_excluded, test_statistics, used_neighborhoods)
            ]

    else:
        raise ValueError(f"Unknown null distribution: {null_distribution}. Use 'neighbors' or 'velocities'.")
    debug_dict['neighborhoods_all'] = neighborhoods

    if cosine_empty_neighborhood is not None:
        test_statistics_velocity = test_statistics[:, 0]
        test_statistics_random = test_statistics[:, 1:]
    else:
        test_statistics_velocity = [test_statistic[0] for test_statistic in test_statistics]
        test_statistics_random = [test_statistic[1:] for test_statistic in test_statistics]
    if cosine_empty_neighborhood is not None:
        p_values_ = p_values(test_statistics_velocity, test_statistics_random).numpy()
    else:
        p_values_ = p_values_list(test_statistics_velocity, test_statistics_random).numpy()

    pvals_corrected = correct_for_multiple_testing(p_values_, correction)
    h0_rejected = pvals_corrected < alpha

    p_values_all = 2 * np.ones(number_cells)
    p_values_all[non_empty_neighborhoods_bool] = p_values_
    if h0_rejected is not None:
        h0_rejected_all = np.zeros(number_cells, dtype=bool)
        h0_rejected_all[non_empty_neighborhoods_bool] = h0_rejected
    else:
        h0_rejected_all = None

    if isinstance(test_statistics_velocity, torch.Tensor):
        test_statistics_velocity = test_statistics_velocity.numpy()
    if isinstance(test_statistics_random, torch.Tensor):
        test_statistics_random = test_statistics_random.numpy()

    debug_dict['test_statistics_velocity'] = test_statistics_velocity
    debug_dict['test_statistics_random'] = test_statistics_random

    return p_values_all, h0_rejected_all, debug_dict


def run_hypothesis_test_on(adata, ekey='Ms', vkey='velocity', basis='umap', restrict_to_velocity_genes=False, **kwargs):
    """
    Runs the hypothesis test using high dimensional expressions, high dimensional velocity,
    and the embeddings from an adata object. For details, see `run_hypothesis_test`.

    :param adata: Anndata object containing high dimensional data and embeddings.
    :param ekey: Name of layer in adata object containing high dimensional expression data.
    :param vkey: Name of layer in adata object containing high dimensional velocity data.
    :param basis: Name of embedding.
    :param restrict_to_velocity_genes: Only use velocity genes determined by the velocity estimation method
        for any high-dimensional computations, specifically cosine similarity.
    :param kwargs: Additional arguments for `run_hypothesis_test`.
    :return: See `run_hypothesis_test`.
    """
    X_expr = adata.layers[ekey]
    X_velo_vector = adata.layers[vkey]
    if restrict_to_velocity_genes:
        print(f"Dropping all the genes which are not velocity genes. {np.sum(adata.var.velocity_genes)} genes left.")
        X_expr = X_expr[:, adata.var.velocity_genes]
        X_velo_vector = X_velo_vector[:, adata.var.velocity_genes]
    Z_expr = adata.obsm[f"X_{basis}"]
    Z_velo_position = Z_expr + adata.obsm[f'velocity_{basis}']

    if Z_expr.shape[1] > 2:
        Z_expr = Z_expr[:, :2]
        Z_velo_position = Z_velo_position[:, :2]
        print("Warning: Your basis has more than two dimensions. "
              "Using only the first two dimensions of the embedding for hypothesis testing like scvelo in "
              "its visualisations.")

    X_expr = torch.tensor(X_expr)
    X_velo_vector = torch.tensor(X_velo_vector)
    Z_expr = torch.tensor(Z_expr)
    Z_velo_position = torch.tensor(Z_velo_position)

    return run_hypothesis_test(X_expr, X_velo_vector, Z_expr, Z_velo_position, **kwargs)
