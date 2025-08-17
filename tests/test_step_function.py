import unittest
import torch
import numpy as np

from velotest.step_function import StepFunction1D


class StepFunction1DTest(unittest.TestCase):
    def test_pdf(self):
        func = StepFunction1D(ranges=torch.tensor([[0, 1], [1, 3]]), values=torch.tensor([0, 1]))
        assert func(torch.tensor([0.5])) == 0
        assert func(torch.tensor([2])) == 1

    def test_violation_sorting(self):
        with self.assertRaises(AssertionError):
            StepFunction1D(ranges=torch.tensor([[1, 3], [0, 1]]), values=torch.tensor([0, 1]))

    def test_violation_overlap(self):
        with self.assertRaises(AssertionError):
            StepFunction1D(ranges=torch.tensor([[0, 2], [1, 3]]), values=torch.tensor([0, 1]))

    def test_get_ranges_beginning(self):
        func = StepFunction1D(ranges=torch.tensor([[0, 1], [1, 2], [2, 3]]), values=torch.tensor([0, 1, 2]))
        exclusion_angle = 0.5
        expected_ranges = torch.tensor([[0.5, 1], [1, 2], [2, 3]])
        assert torch.allclose(func.get_ranges(exclusion_angle), expected_ranges)

    def test_get_ranges_end(self):
        func = StepFunction1D(ranges=torch.tensor([[1, 2.], [2, 6]]), values=torch.tensor([0, 1]))
        exclusion_angle = 0.5
        expected_ranges = torch.tensor([[1, 2], [2, 2 * np.pi-0.5]])
        assert torch.allclose(func.get_ranges(exclusion_angle), expected_ranges)

    def test_get_ranges_complete(self):
        func = StepFunction1D(ranges=torch.tensor([[0, 0.4], [2, 5]]), values=torch.tensor([0, 1]))
        exclusion_angle = 0.5
        expected_ranges = torch.tensor([[2., 5]])
        assert torch.allclose(func.get_ranges(exclusion_angle), expected_ranges)

    def test_get_ranges_None(self):
        func = StepFunction1D(ranges=torch.tensor([[0, 1], [1, 2], [2, 3]]), values=torch.tensor([0, 1, 2]))
        expected_ranges = torch.tensor([[0, 1.], [1, 2], [2, 3]])
        assert torch.allclose(func.get_ranges(), expected_ranges)
