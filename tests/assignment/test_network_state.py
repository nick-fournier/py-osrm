"""Tests for NetworkState."""

import numpy as np
import pytest

from osrm.assignment.network_state import NetworkState


@pytest.fixture
def simple_network():
    """A simple 4-edge network: A→B→C→D (chain) + A→D (shortcut)."""
    return NetworkState.from_edges(
        from_ids=np.array([100, 200, 300, 100], dtype=np.uint64),
        to_ids=np.array([200, 300, 400, 400], dtype=np.uint64),
        lengths_m=np.array([1000.0, 500.0, 800.0, 2000.0]),
        freeflow_kmh=np.array([60.0, 50.0, 40.0, 80.0]),
        jam_density=np.array([150.0, 130.0, 120.0, 150.0]),
        n_lanes=np.array([2, 1, 1, 3], dtype=np.uint8),
    )


class TestConstruction:
    def test_from_edges_shape(self, simple_network):
        net = simple_network
        assert net.n_edges == 4
        assert net.edge_ids.shape == (4, 2)
        assert net.flow_vph.shape == (4,)
        assert net.speed_kmh.shape == (4,)

    def test_initial_speed_is_freeflow(self, simple_network):
        np.testing.assert_array_equal(
            simple_network.speed_kmh, simple_network.freeflow_kmh
        )

    def test_initial_flow_is_zero(self, simple_network):
        np.testing.assert_array_equal(
            simple_network.flow_vph, np.zeros(4)
        )

    def test_empty_network(self):
        net = NetworkState.from_edges(
            from_ids=np.array([], dtype=np.uint64),
            to_ids=np.array([], dtype=np.uint64),
            lengths_m=np.array([], dtype=np.float64),
            freeflow_kmh=np.array([], dtype=np.float64),
            jam_density=np.array([], dtype=np.float64),
            n_lanes=np.array([], dtype=np.uint8),
        )
        assert net.n_edges == 0


class TestEdgeLookup:
    def test_known_edge(self, simple_network):
        assert simple_network.edge_ordinal(100, 200) == 0
        assert simple_network.edge_ordinal(200, 300) == 1
        assert simple_network.edge_ordinal(100, 400) == 3

    def test_unknown_edge(self, simple_network):
        assert simple_network.edge_ordinal(999, 888) is None

    def test_reverse_direction_not_found(self, simple_network):
        """Directed: A→B exists but B→A does not."""
        assert simple_network.edge_ordinal(200, 100) is None


class TestFlowAccumulation:
    def test_accumulate_flow(self, simple_network):
        net = simple_network
        indices = np.array([0, 1, 0])
        increments = np.array([100.0, 200.0, 50.0])
        net.accumulate_flow(indices, increments)
        assert net.flow_vph[0] == 150.0
        assert net.flow_vph[1] == 200.0
        assert net.flow_vph[2] == 0.0

    def test_accumulate_route(self, simple_network):
        net = simple_network
        matched = net.accumulate_route([100, 200, 300, 400], 500.0)
        assert matched == 3  # 3 edges on chain
        assert net.flow_vph[0] == 500.0  # A→B
        assert net.flow_vph[1] == 500.0  # B→C
        assert net.flow_vph[2] == 500.0  # C→D
        assert net.flow_vph[3] == 0.0  # A→D shortcut not used

    def test_accumulate_route_partial_match(self, simple_network):
        """Route with unknown edges still loads known ones."""
        net = simple_network
        matched = net.accumulate_route([100, 200, 999], 300.0)
        assert matched == 1
        assert net.flow_vph[0] == 300.0

    def test_reset_flow(self, simple_network):
        net = simple_network
        net.accumulate_route([100, 200, 300], 500.0)
        net.reset_flow()
        np.testing.assert_array_equal(net.flow_vph, np.zeros(4))


class TestFromRouteAnnotations:
    def test_single_route(self):
        route_result = {
            "routes": [
                {
                    "legs": [
                        {
                            "annotation": {
                                "nodes": [10, 20, 30],
                                "distance": [500.0, 300.0],
                                "speed": [16.67, 11.11],  # m/s
                            }
                        }
                    ]
                }
            ]
        }
        net = NetworkState.from_route_annotations([route_result])
        assert net.n_edges == 2
        assert net.edge_ordinal(10, 20) is not None
        assert net.edge_ordinal(20, 30) is not None

    def test_deduplicates_edges(self):
        """Same edge from two routes should appear once."""
        r1 = {
            "routes": [{"legs": [{"annotation": {
                "nodes": [10, 20], "distance": [100.0], "speed": [10.0]
            }}]}]
        }
        r2 = {
            "routes": [{"legs": [{"annotation": {
                "nodes": [10, 20, 30], "distance": [100.0, 200.0], "speed": [10.0, 15.0]
            }}]}]
        }
        net = NetworkState.from_route_annotations([r1, r2])
        assert net.n_edges == 2  # (10,20) and (20,30)

    def test_empty_routes(self):
        net = NetworkState.from_route_annotations([])
        assert net.n_edges == 0
