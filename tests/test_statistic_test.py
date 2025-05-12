import torch
import scvelo

from velotest.test_statistic import (cos_directionality_one_cell_batch_same_neighbors, \
                                     cos_directionality_one_cell_one_neighborhood,
                                     mean_cos_directionality_varying_neighbors,
                                     mean_cos_directionality_varying_neighborhoods_same_neighbors)


def test_cos_directionality_one_cell_batch_same_neighbors():
    expression = torch.tensor([0, 0, 0.])
    velocity_vector = torch.tensor([1, 0, 0.])
    expressions_neighbours = torch.tensor([[[0, 1, 0], [-1, 0, 0]]])
    cos = cos_directionality_one_cell_batch_same_neighbors(expression, velocity_vector, expressions_neighbours)
    assert torch.allclose(cos, torch.tensor([0, -1.]), atol=1e-6)


def test_cos_directionality_one_cell_one_neighborhood():
    expression = torch.tensor([0, 0, 0.])
    velocity_vector = torch.tensor([1, 0, 0.])
    expressions_neighbours = torch.tensor([[0, 1, 0], [-1, 0, 0]])
    cos = cos_directionality_one_cell_one_neighborhood(expression, velocity_vector, expressions_neighbours)
    assert torch.allclose(cos, torch.tensor([0, -1.]), atol=1e-6)


def test_mean_cos_directionality_varying_neighbors():
    expression = torch.tensor([[0, 0, 0.], [0, 1, 0.], [-1, 0, 0.]])
    velocity_vector = torch.tensor([[1, 0, 0.], [0, 1, 0.], [1, 0, 0.]])
    neighborhoods = [[[], []], [[], []], [[0], [0]]]
    original_indices_cells = [0, 1, 2]
    mean_cos = mean_cos_directionality_varying_neighbors(expression, velocity_vector, neighborhoods,
                                                         original_indices_cells)
    assert torch.allclose(mean_cos, torch.tensor([[1, 1], [1, 1], [1, 1.]]), atol=1e-6)


def test_same_results():
    adata = scvelo.datasets.pancreas()
    adata = adata[:50]
    scvelo.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # Compute velocity
    scvelo.tl.velocity(adata)

    expression = torch.tensor(adata.layers['Ms'])
    velocity_vector = torch.tensor(adata.layers['velocity'])
    original_indices_cells = list(range(50))

    neighborhoods_same = [torch.tensor([list(range(25)), list(range(25, 50))]) for _ in range(50)]
    neighborhoods_varying = [[torch.tensor(list(range(25))), torch.tensor(list(range(25, 50)))] for _ in range(50)]

    result_same_neighbors = mean_cos_directionality_varying_neighborhoods_same_neighbors(expression, velocity_vector,
                                                                                         neighborhoods_same,
                                                                                         original_indices_cells)
    result_varying_neighbors = mean_cos_directionality_varying_neighbors(expression, velocity_vector, neighborhoods_varying,
                                                                         original_indices_cells)
    assert torch.allclose(result_same_neighbors.flatten(), result_varying_neighbors.flatten(), atol=1e-6)
