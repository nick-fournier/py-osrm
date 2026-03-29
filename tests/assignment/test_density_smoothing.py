"""Tests for DensitySmoothing."""

import numpy as np
import pytest

from osrm.assignment.density_smoothing import DensitySmoothing, DensitySmoothingConfig


@pytest.fixture
def chain_3():
    """3-link chain: e0→e1→e2 (sharing nodes)."""
    edge_ids = np.array([
        [10, 20],
        [20, 30],
        [30, 40],
    ], dtype=np.uint64)
    lengths = np.array([1000.0, 1000.0, 1000.0])
    return edge_ids, lengths


class TestIdentity:
    def test_beta_zero_returns_original(self, chain_3):
        """β=0 → no smoothing."""
        edge_ids, lengths = chain_3
        config = DensitySmoothingConfig(beta=0.0, passes=1)
        ds = DensitySmoothing(config)
        ds.build_adjacency(edge_ids, lengths)

        density = np.array([10.0, 50.0, 5.0])
        result = ds.smooth(density)
        np.testing.assert_array_equal(result, density)

    def test_method_none_returns_original(self, chain_3):
        """method='none' skips smoothing."""
        edge_ids, lengths = chain_3
        config = DensitySmoothingConfig(method="none")
        ds = DensitySmoothing(config)
        ds.build_adjacency(edge_ids, lengths)

        density = np.array([10.0, 50.0, 5.0])
        result = ds.smooth(density)
        np.testing.assert_array_equal(result, density)


class TestConservation:
    def test_total_density_preserved_on_regular_graph(self):
        """Smoothing conserves total density on a regular (symmetric) graph.

        On a ring (every node has same degree), the operator is
        doubly-stochastic and total density is preserved.
        """
        # Ring: e0→e1→e2→e0 (each edge shares 2 nodes with 2 neighbors)
        edge_ids = np.array([
            [10, 20],
            [20, 30],
            [30, 10],
        ], dtype=np.uint64)
        lengths = np.array([1000.0, 1000.0, 1000.0])
        config = DensitySmoothingConfig(beta=0.3, passes=3)
        ds = DensitySmoothing(config)
        ds.build_adjacency(edge_ids, lengths)

        density = np.array([100.0, 10.0, 50.0])
        total_before = np.sum(density)
        smoothed = ds.smooth(density)
        total_after = np.sum(smoothed)
        np.testing.assert_allclose(total_after, total_before, rtol=1e-10)


class TestConvergence:
    def test_many_passes_converge_to_uniform_on_ring(self):
        """On a symmetric graph, repeated passes converge to uniform density."""
        edge_ids = np.array([
            [10, 20],
            [20, 30],
            [30, 10],
        ], dtype=np.uint64)
        lengths = np.array([1000.0, 1000.0, 1000.0])
        config = DensitySmoothingConfig(beta=0.4, passes=200)
        ds = DensitySmoothing(config)
        ds.build_adjacency(edge_ids, lengths)

        density = np.array([100.0, 0.0, 0.0])
        result = ds.smooth(density)
        np.testing.assert_allclose(
            result, np.full(3, density.mean()), atol=1.0
        )

    def test_many_passes_reduce_variance(self, chain_3):
        """On any graph, repeated passes should reduce density variance."""
        edge_ids, lengths = chain_3
        config = DensitySmoothingConfig(beta=0.3, passes=50)
        ds = DensitySmoothing(config)
        ds.build_adjacency(edge_ids, lengths)

        density = np.array([100.0, 0.0, 0.0])
        result = ds.smooth(density)
        assert np.var(result) < np.var(density)


class TestKnownTopology:
    def test_3link_chain_one_pass(self, chain_3):
        """Hand-calculated: 3-link chain, equal lengths, β=0.5.

        Adjacency (row-normalized, equal lengths):
        e0 neighbors: [e1] (share node 20)       → W[0,1] = 1.0
        e1 neighbors: [e0, e2] (share 20 and 30) → W[1,0] = 0.5, W[1,2] = 0.5
        e2 neighbors: [e1] (share node 30)       → W[2,1] = 1.0

        k = [10, 50, 5]
        k̃_0 = 0.5*10 + 0.5*50       = 30.0
        k̃_1 = 0.5*50 + 0.5*(0.5*10 + 0.5*5) = 25 + 3.75 = 28.75
        k̃_2 = 0.5*5  + 0.5*50       = 27.5
        """
        edge_ids, lengths = chain_3
        config = DensitySmoothingConfig(beta=0.5, passes=1)
        ds = DensitySmoothing(config)
        ds.build_adjacency(edge_ids, lengths)

        density = np.array([10.0, 50.0, 5.0])
        result = ds.smooth(density)
        np.testing.assert_allclose(result, [30.0, 28.75, 27.5], atol=1e-10)


class TestEmptyNetwork:
    def test_empty_edges(self):
        config = DensitySmoothingConfig(beta=0.3)
        ds = DensitySmoothing(config)
        ds.build_adjacency(
            np.empty((0, 2), dtype=np.uint64),
            np.empty(0, dtype=np.float64),
        )
        result = ds.smooth(np.empty(0))
        assert len(result) == 0


class TestDisconnected:
    def test_disconnected_edges_no_smoothing(self):
        """Edges with no shared nodes should not affect each other."""
        edge_ids = np.array([
            [10, 20],
            [30, 40],
        ], dtype=np.uint64)
        lengths = np.array([1000.0, 1000.0])
        config = DensitySmoothingConfig(beta=0.5, passes=1)
        ds = DensitySmoothing(config)
        ds.build_adjacency(edge_ids, lengths)

        density = np.array([100.0, 0.0])
        result = ds.smooth(density)
        np.testing.assert_array_equal(result, density)
