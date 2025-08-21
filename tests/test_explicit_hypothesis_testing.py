import unittest

import torch

from velotest.explicit_hypothesis_testing import compute_position_on_unit_circle


class TestExplicitHypothesisTesting(unittest.TestCase):
    def test_compute_position_on_unit_circle(self):
        Z_expr = torch.tensor([[0, 1], [0, 2], [1, 1], [0, 0], [-1, 1]], dtype=torch.float32)
        Z_velo_vector = torch.tensor([[0, 1], [-1, 0], [-1, 0], [-1, -1], [0, -1]], dtype=torch.float32)
        nn_indices = torch.tensor([[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]], dtype=torch.long)

        aligned_theta, _ = compute_position_on_unit_circle(Z_expr, Z_velo_vector, nn_indices)
        assert torch.allclose(aligned_theta[0],
                              torch.tensor([0, 1.5 * torch.pi, torch.pi, 0.5 * torch.pi], dtype=torch.float32),
                              atol=1e-3)
