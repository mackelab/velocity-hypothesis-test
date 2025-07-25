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
        used_neighborhoods = None
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


def _process_task(task: Tuple[int, int, int, List[int]]
                 ) -> Tuple[int, int, torch.Tensor]:
    """
    Process one valid (non‑empty) neighborhood task.
    Returns (cell_idx, neigh_id, computed_mean_cosine).
    """
    cell_idx, orig_idx, neigh_id, neigh = task
    m = torch.mean(
        cos_directionality_one_cell_one_neighborhood(
            _expr[orig_idx],
            _vel[orig_idx],
            _expr[neigh]
        )
    )
    assert not torch.isnan(m), "NaN in cosine!"
    return cell_idx, neigh_id, m


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
    per_cell_vals: List[dict] = [{} for _ in range(n_cells)]

    # build tasks only for non‑empty neighborhoods
    tasks: List[Tuple[int, int, int, torch.Tensor]] = []
    for cell_idx, orig_idx in enumerate(original_indices_cells):
        for neigh_id, neigh in enumerate(neighborhoods[cell_idx]):
            if len(neigh) > 0:  # only non‑empty
                tasks.append((cell_idx, orig_idx, neigh_id, neigh))

    # parallel map
    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(expression, velocity_vector)
    ) as pool:
        for cell_idx, neigh_id, val in tqdm(pool.imap_unordered(_process_task, tasks, chunksize=200000), total=len(tasks)):
            used_mask[cell_idx, neigh_id] = True
            # Make sure that vals in the right order based on neigh_id
            per_cell_vals[cell_idx][neigh_id] = val

    mean_cos_neighborhoods: List[torch.Tensor] = []

    for cell_idx, cell_dict in enumerate(per_cell_vals):
        # get the sorted neighborhood IDs that were actually computed
        sorted_neigh_ids = sorted(cell_dict.keys())
        # pull the values in order
        vals_in_order = [cell_dict[nid] for nid in sorted_neigh_ids]
        # convert to tensor
        mean_cos_neighborhoods.append(
            torch.tensor(vals_in_order, dtype=torch.float32)
        )

    return mean_cos_neighborhoods, used_mask
