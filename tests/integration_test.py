import unittest

import anndata
import numpy as np
import scanpy
import scvelo
import torch
from parameterized import parameterized

from velotest.hypothesis_testing import run_hypothesis_test_on


class IntegrationTest(unittest.TestCase):
    __adata: anndata.AnnData

    @classmethod
    def setUpClass(cls):
        # Load data
        adata = scvelo.datasets.pancreas()
        adata = adata[:50]
        scvelo.pp.filter_genes(adata, min_shared_counts=20)
        scvelo.pp.normalize_per_cell(adata)
        scvelo.pp.filter_genes_dispersion(adata, n_top_genes=2000)
        scanpy.pp.log1p(adata)
        scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)

        # Compute velocity
        scvelo.tl.velocity(adata)

        # Compute 2D embedding of velocity vectors
        scvelo.tl.velocity_graph(adata)
        scvelo.tl.velocity_embedding(adata)
        cls.__adata = adata

    @parameterized.expand([['neighbors', 2, None], ['neighbors', 2, 10], ['velocities', None, None],
                           ['velocities', None, 10], ['velocities', 2, None], ['velocities', 2, 10],
                           ['velocities-explicit', None, None], ['velocities-explicit', None, 10]])
    def test_run_hypothesis_test_on_pancreas(self, null_distribution, cosine_empty_neighborhood, exclusion_degree):
        adata = self.__adata.copy()

        # Run test
        uncorrected_p_values, h0_rejected, _ = run_hypothesis_test_on(adata, number_neighborhoods=100,
                                                                      number_neighbors_to_sample_from=20,
                                                                      null_distribution=null_distribution,
                                                                      cosine_empty_neighborhood=cosine_empty_neighborhood,
                                                                      exclusion_degree=exclusion_degree)
        assert uncorrected_p_values.shape[0] == adata.n_obs
        assert h0_rejected.shape[0] == adata.n_obs

    @parameterized.expand(["integer", "np_array", "tensor"])
    def test_run_hypothesis_test_on_pancreas_variable_k(self, k_mode):
        adata = self.__adata.copy()
        n_cells = adata.n_obs

        if k_mode == "integer":
            number_neighbors_to_sample_from = 10
        elif k_mode == "np_array":
            number_neighbors_to_sample_from = np.repeat([5, 4], [n_cells // 2, n_cells - n_cells // 2])
        elif k_mode == "tensor":
            number_neighbors_to_sample_from = torch.tensor(np.repeat([5, 4], [n_cells // 2, n_cells - n_cells // 2]))

        # Run test
        uncorrected_p_values, h0_rejected, _ = run_hypothesis_test_on(adata, number_neighbors_to_sample_from=number_neighbors_to_sample_from)
        assert uncorrected_p_values.shape[0] == adata.n_obs
        assert h0_rejected.shape[0] == adata.n_obs

    def test_run_hypothesis_test_on_pancreas_variable_k_same_results(self):
        adata = self.__adata.copy()

        number_neighbors_to_sample_from_int = 10
        number_neighbors_to_sample_from_array = np.repeat([10], adata.n_obs)

        uncorrected_p_values_int, h0_rejected_int, _ = run_hypothesis_test_on(adata, number_neighbors_to_sample_from=number_neighbors_to_sample_from_int)
        uncorrected_p_values_array, h0_rejected_array, _ = run_hypothesis_test_on(adata, number_neighbors_to_sample_from=number_neighbors_to_sample_from_array)

        np.testing.assert_array_equal(uncorrected_p_values_int, uncorrected_p_values_array)
        np.testing.assert_array_equal(h0_rejected_int, h0_rejected_array)

    def test_run_hypothesis_test_on_pancreas_variable_k_same_results_different_values(self):
        adata = self.__adata.copy()
        n_cells = adata.n_obs

        value_0 = 10
        value_1 = 15

        number_neighbors_to_sample_from_array = np.repeat([value_0, value_1], [n_cells // 2, n_cells - n_cells // 2])

        uncorrected_p_values_0, h0_rejected_0, _ = run_hypothesis_test_on(adata,
                                                                          number_neighbors_to_sample_from=value_0)
        uncorrected_p_values_1, h0_rejected_1, _ = run_hypothesis_test_on(adata,
                                                                          number_neighbors_to_sample_from=value_1)
        uncorrected_p_values_combined = np.concatenate([uncorrected_p_values_0[:n_cells // 2], uncorrected_p_values_1[n_cells - n_cells // 2:]])
        h0_rejected_combined = np.concatenate([h0_rejected_0[:n_cells // 2], h0_rejected_1[n_cells - n_cells // 2:]])

        uncorrected_p_values_array, h0_rejected_array, _ = run_hypothesis_test_on(adata, number_neighbors_to_sample_from=number_neighbors_to_sample_from_array)

        np.testing.assert_array_equal(uncorrected_p_values_combined, uncorrected_p_values_array)
        np.testing.assert_array_equal(h0_rejected_combined, h0_rejected_array)
