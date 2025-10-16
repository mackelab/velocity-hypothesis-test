from typing import Optional

import torch
from numpy import ndarray
from tqdm import tqdm


def cos_directionality_one_cell_batch_same_neighbors(expression: torch.Tensor, velocity_vector: torch.Tensor,
                                                     expressions_neighbours: torch.Tensor):
    """
    Calculates the cosine similarity between the velocity of a cell and multiple sets of other cells
    (e.g., in the neighborhood of the cell). Every set is assumed to have same number of neighbors.

    :param expression: vector of gene expressions of the cell
    :param velocity_vector: velocity vector of the cell, not the position (x+v) of the velocity
    :param expressions_neighbours: (#neighborhoods, #neighbors, #genes)
    :return:
    """
    number_neighborhoods = expressions_neighbours.shape[0]
    number_neighbors_per_neighborhood = expressions_neighbours.shape[1]

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    return cos(expressions_neighbours - expression,
               velocity_vector[None, None, :].expand(number_neighborhoods,
                                                     number_neighbors_per_neighborhood, -1))


def cos_directionality_one_cell_one_neighborhood(expression: torch.Tensor, velocity_vector: torch.Tensor,
                                                 expressions_neighbours: torch.Tensor):
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


def mean_cos_directionality_varying_neighborhoods_same_neighbors(expression: torch.Tensor,
                                                                 velocity_vector: torch.Tensor,
                                                                 neighborhoods: list, original_indices_cells):
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
    number_cells = len(neighborhoods)
    number_neighborhoods = neighborhoods[0].shape[0]
    mean_cos_neighborhoods = torch.zeros((number_cells, number_neighborhoods))
    for i, original_index in enumerate(tqdm(original_indices_cells)):
        mean_cos_neighborhoods[i] = torch.mean(
            cos_directionality_one_cell_batch_same_neighbors(expression[original_index],
                                                             velocity_vector[original_index],
                                                             expression[neighborhoods[i]]), dim=-1)
    return mean_cos_neighborhoods


def mean_cos_directionality_varying_neighbors(expression: torch.Tensor,
                                              velocity_vector: torch.Tensor,
                                              neighborhoods: list,
                                              original_indices_cells,
                                              cosine_empty_neighborhood: Optional[float] = 2):
    """
    Mean cos directionality for a varying number of neighbors in the neighborhoods across cells:
    Calculates the cosine similarity between the velocity of a cell and multiple sets of other cells
    (e.g., in the neighborhood of the cell). Every set can have a different number of cells.

    :param expression: expression of all cells
    :param velocity_vector: velocity vectors of all cells, not position (x+v) of the velocity
    :param neighborhoods: Neighborhoods of selected cells. list of length #cells with lists of varying #neighbors.
    :param original_indices_cells: indices of the selected cells in the original expression matrix
    :param cosine_empty_neighborhood: if the neighborhood is empty, assign this value to the mean cosine similarity.
        Standard is 2 which is higher then the max of the cosine similarity and will therefore lead to more cells
        where we cannot reject the null hypothesis (Type II error). -2 would lead to Type I errors.
        "None" will ignore empty neighborhoods and then return a variable number of mean cosine similarities per cell.
    :return:
    """
    if torch.cuda.is_available():
        expression = expression.cuda()
        velocity_vector = velocity_vector.cuda()

    number_cells = len(neighborhoods)
    number_neighborhoods = len(neighborhoods[0])
    if cosine_empty_neighborhood is not None:
        mean_cos_neighborhoods = torch.zeros((number_cells, number_neighborhoods))
        used_neighborhoods = torch.ones((number_cells, number_neighborhoods), dtype=torch.bool)
    else:
        mean_cos_neighborhoods = []
        used_neighborhoods = torch.ones((number_cells, number_neighborhoods), dtype=torch.bool)
    for cell, (original_index, neighborhoods_one_cell) in enumerate(zip(tqdm(original_indices_cells), neighborhoods)):
        if cosine_empty_neighborhood is None:
            mean_cos_neighborhoods_cell = []
        for neighborhood_id, neighborhood in enumerate(neighborhoods_one_cell):
            if len(neighborhood) == 0:
                if cosine_empty_neighborhood is not None:
                    mean_cos_neighborhoods[cell, neighborhood_id] = cosine_empty_neighborhood
                else:
                    used_neighborhoods[cell, neighborhood_id] = False
            else:
                mean_cos_directionality_one_cell_one_neighborhood = torch.mean(
                    cos_directionality_one_cell_one_neighborhood(expression[original_index],
                                                                 velocity_vector[original_index],
                                                                 expression[neighborhood]))
                assert not torch.isnan(mean_cos_directionality_one_cell_one_neighborhood), \
                    "Something went wrong and some of the test statistics are NaN. This shouldn't happen."
                if cosine_empty_neighborhood is not None:
                    mean_cos_neighborhoods[cell, neighborhood_id] = mean_cos_directionality_one_cell_one_neighborhood
                else:
                    mean_cos_neighborhoods_cell.append(mean_cos_directionality_one_cell_one_neighborhood)
        if cosine_empty_neighborhood is None:
            mean_cos_neighborhoods.append(torch.tensor(mean_cos_neighborhoods_cell))
    return mean_cos_neighborhoods, used_neighborhoods


##### Parallel option


from multiprocessing import Pool, cpu_count
from typing import List, Tuple

# module‑level globals in each worker
_expr = None
_vel = None


def _init_worker(expr: torch.Tensor,
                 vel: torch.Tensor):
    global _expr, _vel
    _expr = expr
    _vel = vel


def _process_one_cell(args: Tuple[int, int, List[torch.Tensor]]
                     ) -> Tuple[int, List[float], List[bool]]:
    """
    For cell_idx:
      - compute all its non‑empty neighborhoods in one pass
      - return list of floats + list of used‐flags
    """
    cell_idx, orig_idx, neighs = args
    vals, used = [], []
    x_i = _expr[orig_idx]
    v_i = _vel[orig_idx]
    for nb in neighs:
        if len(nb) == 0:
            used.append(False)
        else:
            # vectorized: get all neighbors, compute cosines at once
            e_nb = _expr[nb]                     # (n_nb, genes)
            sims = torch.nn.functional.cosine_similarity(
                        e_nb - x_i,
                        v_i.unsqueeze(0).expand_as(e_nb),
                        dim=1
                   )
            vals.append(sims.mean().item())
            used.append(True)
    return cell_idx, vals, used


def mean_cos_directionality_varying_neighbors_parallel(
    expression: torch.Tensor,
    velocity_vector: torch.Tensor,
    neighborhoods: List[List[torch.Tensor]],
    original_indices_cells: ndarray,
    n_workers: int = None
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Returns:
      - mean_cos_neighborhoods: List[Tensor] per cell, only for non‑empty neighbors
      - used_mask: (n_cells, n_neigh) bool Tensor marking non‑empty slots
    """

    n_cells = len(neighborhoods)
    n_neigh = len(neighborhoods[0])
    n_workers = n_workers or cpu_count()-1

    # prepare outputs
    used_mask = torch.zeros((n_cells, n_neigh), dtype=torch.bool)
    per_cell_vals: List[List[float]] = [[] for _ in range(n_cells)]

    # build one task per cell
    tasks = [
        (cell_idx, orig_idx, neighborhoods[cell_idx])
        for cell_idx, orig_idx in enumerate(original_indices_cells)
    ]

    # parallel map
    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(expression, velocity_vector)
    ) as pool:
        for cell_idx, vals, used in tqdm(pool.imap_unordered(_process_one_cell, tasks), total=n_cells):
            used_mask[cell_idx, :] = torch.tensor(used, dtype=torch.bool)
            per_cell_vals[cell_idx] = vals

    mean_cos_neighborhoods: List[torch.Tensor] = []

    mean_cos_neighborhoods = [
        torch.tensor(vals, dtype=torch.float32)
        for vals in per_cell_vals
    ]

    return mean_cos_neighborhoods, used_mask


##### torch option

def mean_cos_directionality_varying_neighbors_torch(
        expression: torch.Tensor,  # (N_all_cells, G)
        velocity_vector: torch.Tensor,  # (N_all_cells, G)
        neighborhoods: List[List[List[int]]],  # len = N_selected_cells, each inner list len = N_neigh
        original_indices_cells: List[int],  # len = N_selected_cells
        chunk_size: int = 100000
) -> Tuple[
    List[torch.Tensor],  # per‑cell list of 1D tensors
    torch.BoolTensor,  # used_mask:       (N_cells, N_neigh)
]:
    # 1) Gather the selected cells’ expressions & velocities
    expr_sel = expression[original_indices_cells]  # (C, G)
    vel_sel = velocity_vector[original_indices_cells]  # (C, G)

    C, G = expr_sel.shape
    N_neigh = len(neighborhoods[0])

    # 2) Flatten all non‐empty (cell, neigh_id, each neighbor index) into 3 vectors
    cell_ids = []
    neigh_ids = []
    nbr_indices = []
    for i in range(C):
        for j in range(N_neigh):
            nbrs = neighborhoods[i][j]
            if len(nbrs) > 0:  # skip empties entirely
                cell_ids.extend([i] * len(nbrs))
                neigh_ids.extend([j] * len(nbrs))
                nbr_indices.extend(nbrs)

    if len(nbr_indices) == 0:
        # edge case: all neighborhoods empty
        mean_cos_matrix = torch.zeros((C, N_neigh), dtype=torch.float32)
        used_mask = torch.zeros((C, N_neigh), dtype=torch.bool)
        mean_cos_list = [torch.empty(0) for _ in range(C)]
        return mean_cos_list, used_mask

    cell_ids = torch.tensor(cell_ids, dtype=torch.long)
    neigh_ids = torch.tensor(neigh_ids, dtype=torch.long)
    nbr_indices = torch.tensor(nbr_indices, dtype=torch.long)

    # 2) Setup accumulators
    T = C * N_neigh
    sums = torch.zeros((T,), dtype=torch.float32)
    counts = torch.zeros((T,), dtype=torch.float32)

    # 3) Process in chunks
    M = nbr_indices.shape[0]
    for start in tqdm(range(0, M, chunk_size), total=(M + chunk_size - 1) // chunk_size):
        end = min(start + chunk_size, M)
        idx = slice(start, end)

        # pull only this chunk’s neighbors & owner cells
        ni = nbr_indices[idx]  # (B,)
        ci = cell_ids[idx]  # (B,)
        hi = neigh_ids[idx]  # (B,)

        # gather expressions & vel
        en = expression[ni]  # (B, G)
        ec = expr_sel[ci]  # (B, G)
        vc = vel_sel[ci]  # (B, G)

        # compute cosines
        deltas = en - ec  # (B, G)
        cosines = torch.nn.functional.cosine_similarity(deltas, vc, dim=1, eps=1e-6)  # (B,)

        # flat group ids = cell * N_neigh + neigh
        gids = ci * N_neigh + hi  # (B,)

        # scatter‑add into the big buffers
        if cosines.dtype != torch.float32:
            cosines = cosines.to(torch.float32)
        sums.scatter_add_(0, gids, cosines)
        counts.scatter_add_(0, gids, torch.ones_like(cosines))

    # 4) Finalize
    means_flat = sums / counts.clamp(min=1.0)
    mean_cos_matrix = means_flat.view(C, N_neigh)
    used_mask = (counts.view(C, N_neigh) > 0)

    # 7) Build the per‑cell list of 1D tensors
    mean_cos_list: List[torch.Tensor] = []
    for i in range(C):
        valid_means = mean_cos_matrix[i][used_mask[i]]
        mean_cos_list.append(valid_means)

    return mean_cos_list, used_mask
