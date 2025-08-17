import logging
from collections import deque
from multiprocessing import cpu_count, Pool

import numpy as np
import torch
from tqdm import tqdm

from velotest.neighbors import find_neighbors
from velotest.test_statistic import cos_directionality_one_cell_one_neighborhood
from velotest.test_statistic_function import TestStatistic

logging.basicConfig(level=logging.ERROR)


def set_subtraction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return elements in a which are not in b. 1D tensors."""
    combined = torch.cat((a, b))
    uniques, counts = combined.unique(return_counts=True)
    return uniques[counts == 1]


def project_points_to_unit_circle(Z_expr_cell, Z_expr_neighbors):
    position_centered = Z_expr_neighbors - Z_expr_cell
    theta = torch.arctan2(position_centered[:, 1], position_centered[:, 0])
    cartesian = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    return theta, cartesian


def run_explicit_test_from(adata, **kwargs):
    """
    Run the hypothesis test on the given AnnData object.

    :param adata: AnnData object containing the data.
    :param kwargs: Additional parameters for the test.
    :return: List of TestStatistic objects for each cell.
    """
    X_expr = torch.tensor(adata.layers['Ms'], dtype=torch.float32)
    X_velo_vector = torch.tensor(adata.layers['velocity'], dtype=torch.float32)
    Z_expr = torch.tensor(adata.obsm['X_umap'], dtype=torch.float32)
    Z_velo_position = Z_expr + torch.tensor(adata.obsm['velocity_umap'], dtype=torch.float32)

    return run_explicit_test(X_expr, X_velo_vector, Z_expr, Z_velo_position, **kwargs)


def run_explicit_test(X_expr, X_velo_vector, Z_expr, Z_velo_position, number_neighbors_to_sample_from=300,
                      gamma_deg: float = 22.5, parallel: bool = True, exclusion_deg: float = None):
    if not isinstance(X_expr, torch.Tensor):
        X_expr = torch.tensor(X_expr, dtype=torch.float32)
    if not isinstance(X_velo_vector, torch.Tensor):
        X_velo_vector = torch.tensor(X_velo_vector, dtype=torch.float32)
    if not isinstance(Z_expr, torch.Tensor):
        Z_expr = torch.tensor(Z_expr, dtype=torch.float32)
    if not isinstance(Z_velo_position, torch.Tensor):
        Z_velo_position = torch.tensor(Z_velo_position, dtype=torch.float32)

    Z_velo_vector = Z_velo_position - Z_expr
    import time
    starttime = time.time()
    nn_indices = find_neighbors(Z_expr, k_neighbors=number_neighbors_to_sample_from)
    nn_indices = torch.tensor(nn_indices, dtype=torch.long)
    logging.debug(f"Found neighbors in {time.time() - starttime:.3f} seconds")

    starttime = time.time()
    cos_directionality_one_cell_one_neighborhood_batched = torch.vmap(cos_directionality_one_cell_one_neighborhood,
                                                                      chunk_size=100)
    cos_sim = cos_directionality_one_cell_one_neighborhood_batched(X_expr, X_velo_vector, X_expr[nn_indices])
    logging.debug(f"Computed cosine similarity in {time.time() - starttime:.3f} seconds")

    aligned_theta = compute_position_on_unit_circle(Z_expr, Z_velo_vector, nn_indices)

    if not parallel:
        times = []
        results = []
        for cell in tqdm(range(Z_expr.shape[0])):
            starttime = time.time()
            ranges, values = compute_step_statistics(cell, cos_sim, aligned_theta, np.deg2rad(gamma_deg))
            results.append((ranges, values))
            elapsed_time = time.time() - starttime
            times.append(elapsed_time)
        logging.debug(f"Mean time per cell: {np.mean(times):.3f} seconds")
    else:
        num_cells = Z_expr.shape[0]
        gamma = float(np.deg2rad(gamma_deg))

        # Use initializer so cos_sim and aligned_theta are copied to each worker only once
        with Pool(processes=cpu_count() - 1, initializer=_init_worker,
                  initargs=(cos_sim, aligned_theta, gamma)) as pool:
            # pool.imap yields results in order; tqdm wraps it to show progress
            results_iter = pool.imap(_worker_cell, range(num_cells), chunksize=20)
            results = list(tqdm(results_iter, total=num_cells))

    starttime = time.time()
    statistics = []
    for ranges, values in results:
        if ranges is not None or values is not None:
            statistics.append(TestStatistic(ranges=np.array(ranges), values=np.array(values)))
        else:
            statistics.append(None)
    logging.debug(f"Created statistic objects in {time.time() - starttime:.3f} seconds")

    starttime = time.time()
    if exclusion_deg is not None:
        exclusion_rad = np.deg2rad(exclusion_deg)
    else:
        exclusion_rad = None
    p_values_uncorrected = np.array([
        statistic.p_value(t_obs=statistic(np.array([0.])), exclusion_angle=exclusion_rad)
        if statistic is not None else 2
        for statistic in statistics])
    logging.debug(f"Computed uncorrected p-values in {time.time() - starttime:.3f} seconds")

    return p_values_uncorrected, statistics


def compute_position_on_unit_circle(Z_expr, Z_velo_vector, nn_indices):
    project_points_to_unit_circle_batched = torch.vmap(project_points_to_unit_circle, chunk_size=100)
    theta, cartesian = project_points_to_unit_circle_batched(Z_expr, Z_expr[nn_indices])
    theta_visualised_velocity = torch.arctan2(Z_velo_vector[:, 1], Z_velo_vector[:, 0])
    aligned_theta = theta - theta_visualised_velocity.unsqueeze(1)
    aligned_theta = aligned_theta % (2 * np.pi)  # Ensure angles are in [0, 2pi)
    return aligned_theta


def compute_step_statistics(cell: int, cos_sim: torch.Tensor, aligned_theta: torch.Tensor,
                            gamma: float) -> (list, list):
    """
    Compute the test statistic and every point where it changes for a given cell.

    :param cell: Index of the cell to compute statistics for.
    :param cos_sim: Cosine similarity values for the high-dimensional velocities.
    :param aligned_theta: Tensor of angles aligned with visualised velocity.
    :param gamma: Angle defining the cone size [in rad].
    :return: Ranges and values for the test statistic.
    """
    # Ensure aligned_theta and cos_sim are tensors
    if not isinstance(aligned_theta, torch.Tensor):
        raise TypeError("aligned_theta must be a torch.Tensor")
    if not isinstance(cos_sim, torch.Tensor):
        raise TypeError("cos_sim must be a torch.Tensor")

    assert torch.all(aligned_theta >= 0) and torch.all(aligned_theta <= 2 * np.pi), \
        "aligned_theta must be in the range [0, 2pi]"
    assert cell <= cos_sim.shape[0], \
        f"Cell index {cell} is out of bounds for cos_sim with shape {cos_sim.shape}"
    assert 0 <= gamma <= np.pi, \
        f"Gamma must be in the range [0, pi], got {gamma}"

    # Initial cone: points in 0 to gamma and 2pi-gamma to 2pi
    position_velocity = 0
    lower_limit_cone = 2 * np.pi - gamma
    upper_limit_cone = gamma
    neighbors_in_cone_bool = torch.logical_or(aligned_theta[cell] > lower_limit_cone,
                                              aligned_theta[cell] < upper_limit_cone)
    neighbors_in_cone = torch.where(neighbors_in_cone_bool)[0]
    if len(neighbors_in_cone) == 0:
        # If no neighbors are in the cone, return Nones ranges and values
        return None, None
    neighbors_outside_cone = set_subtraction(torch.arange(0, aligned_theta.shape[1], 1), neighbors_in_cone)

    indices_sorted_aligned_cell = torch.argsort(aligned_theta[cell])
    points_in_cone_queue = deque([index for index in indices_sorted_aligned_cell if index in neighbors_in_cone])
    # Rotate the queue so that the first point is the one closest to the lower limit of the cone
    points_in_cone_queue.rotate(
        -torch.argmin(
            (aligned_theta[cell][torch.tensor(points_in_cone_queue)] - lower_limit_cone) % (2 * torch.pi)).item())
    points_outside_cone_queue = deque(
        [index for index in indices_sorted_aligned_cell if index in neighbors_outside_cone])
    # Rotate the queue so that the first point is the one closest to the upper limit of the cone
    points_outside_cone_queue.rotate(
        -torch.argmin(
            (aligned_theta[cell][torch.tensor(points_outside_cone_queue)] - upper_limit_cone) % (2 * torch.pi)).item())

    ranges = []
    values = []
    while position_velocity < 2 * np.pi:
        if len(points_in_cone_queue) > 0:
            distance_next_point_in_cone = (aligned_theta[cell][points_in_cone_queue[0]] - lower_limit_cone) % (
                    2 * np.pi)
        else:
            distance_next_point_in_cone = np.inf
        if len(points_outside_cone_queue) > 0:
            distance_next_point_outside_cone = (aligned_theta[cell][
                                                    points_outside_cone_queue[0]] - upper_limit_cone) % (
                                                       2 * np.pi)
        else:
            distance_next_point_outside_cone = np.inf

        point_outside_closer = np.argmin([distance_next_point_in_cone, distance_next_point_outside_cone])

        if point_outside_closer:
            distance = distance_next_point_outside_cone
        else:
            distance = distance_next_point_in_cone

        distance = distance.item()
        lower_limit_cone += distance
        lower_limit_cone %= 2 * torch.pi
        upper_limit_cone += distance
        upper_limit_cone %= 2 * torch.pi

        # Only add range if there are points in the cone, function shouldn't be defined otherwise
        if len(points_in_cone_queue) > 0:
            values.append(torch.mean(cos_sim[cell][torch.tensor(points_in_cone_queue)]).item())
            ranges.append([position_velocity, np.min([position_velocity + distance, 2 * np.pi])])
        position_velocity += distance

        if point_outside_closer:
            # Remove the point outside the cone from the queue
            point = points_outside_cone_queue.popleft()
            # Add the point to the cone
            points_in_cone_queue.append(point)
            # assert (aligned_theta[cell][points_in_cone_queue[-1]] - aligned_theta[cell][points_in_cone_queue[0]]) % (2 * np.pi) <= 2*gamma+1e-4, \
            #    f"Points in cone are not within the cone size, {aligned_theta[cell][points_in_cone_queue[-1]]:.2f} - {aligned_theta[cell][points_in_cone_queue[0]]:.2f} = {(aligned_theta[cell][points_in_cone_queue[-1]] - aligned_theta[cell][points_in_cone_queue[0]]) % (2 * np.pi)} > {2*gamma}. {position_velocity=:.2f}, {distance=:.2f}, {lower_limit_cone=:.2f}, {upper_limit_cone=:.2f}"
        else:
            # Remove the point inside the cone from the queue
            point = points_in_cone_queue.popleft()
            # Add the point outside the cone
            points_outside_cone_queue.append(point)
    return ranges, values


# module-global placeholders for worker processes
_WORKER_COS_SIM = None
_WORKER_ALIGNED_THETA = None
_WORKER_GAMMA = None


def _init_worker(cos_sim_, aligned_theta_, gamma_):
    """Initializer run once in each worker process to store large arrays as globals."""
    global _WORKER_COS_SIM, _WORKER_ALIGNED_THETA, _WORKER_GAMMA
    _WORKER_COS_SIM = cos_sim_
    _WORKER_ALIGNED_THETA = aligned_theta_
    _WORKER_GAMMA = gamma_


def _worker_cell(cell_idx):
    """Worker-callable: computes (ranges, values) for one cell using module-global data."""
    # compute_step_statistics returns (ranges, values) or (None, None)
    return compute_step_statistics(cell_idx, _WORKER_COS_SIM, _WORKER_ALIGNED_THETA, _WORKER_GAMMA)
