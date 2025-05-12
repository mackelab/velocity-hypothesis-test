from velotest.hypothesis_testing import run_hypothesis_test_on
import scvelo
import numpy as np
import pytest


@pytest.mark.parametrize("null_distribution", ('neighbors', 'velocities'))
def test_run_hypothesis_test_on_pancreas(null_distribution):
    # Load data
    adata = scvelo.datasets.pancreas()
    adata = adata[:50]
    scvelo.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # Compute velocity
    scvelo.tl.velocity(adata)

    # Compute 2D embedding of velocity vectors
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_embedding(adata)

    # Run test
    uncorrected_p_values, h0_rejected, _, _, _ = run_hypothesis_test_on(adata, number_neighborhoods=100,
                                                                        number_neighbors_to_sample_from=20,
                                                                        null_distribution=null_distribution)
    assert uncorrected_p_values.shape[0] == adata.n_obs
    assert h0_rejected.shape[0] == adata.n_obs
    if null_distribution == 'neighbors':
        assert np.sum(h0_rejected) >= 1
