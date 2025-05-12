import pytest
import torch

from velotest.neighbors import find_neighbors_in_direction_of_velocity


@pytest.mark.parametrize("threshold_degree", (1, 5, 22.5, 90))
def test_find_neighbors_in_direction_of_velocity_limited_neighborhoods(threshold_degree):
    Z_expr = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1.]])
    Z_velo = torch.tensor([[-1, 1], [-1, -1], [1, -1], [1, 1.]])
    Z_velo_position = Z_expr + Z_velo
    nn_indices = torch.tensor([[1], [2], [3], [0]])

    neighbors = find_neighbors_in_direction_of_velocity(Z_expr, Z_velo_position, nn_indices, threshold_degree)
    assert len(neighbors) == 4
    assert neighbors == [torch.tensor([1]), torch.tensor([2]), torch.tensor([3]),
                         torch.tensor([0])]  # All neighbors are in the direction of the velocity vector


@pytest.mark.parametrize("threshold_degree", (22.5, 40))
def test_find_neighbors_in_direction_of_velocity_complete_neighborhoods(threshold_degree):
    Z_expr = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1.]])
    Z_velo = torch.tensor([[-1, 1], [-1, -1], [1, -1], [1, 1.]])
    Z_velo_position = Z_expr + Z_velo
    nn_indices = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [1, 2, 0]])

    neighbors = find_neighbors_in_direction_of_velocity(Z_expr, Z_velo_position, nn_indices, threshold_degree)
    assert len(neighbors) == 4
    assert neighbors == [torch.tensor([1]), torch.tensor([2]), torch.tensor([3]),
                         torch.tensor([0])]  # All neighbors are in the direction of the velocity vector
