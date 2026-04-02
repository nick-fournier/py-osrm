import types

import numpy as np

from osrm.assignment import (
    AssignmentConfig,
    AssignmentLoop,
    DemandTrip,
    HillClimberResult,
    MatrixAssignmentSolver,
    MatrixFreeHillClimber,
    NetworkState,
    ODMatrixAdapter,
    TripStreamAdapter,
)
from osrm.assignment.solvers import ODLedgerEntry, RouteAssignment


def _trip(dep_s: float, volume: float = 1.0) -> DemandTrip:
    return DemandTrip(
        origin=(7.41, 43.73),
        destination=(7.42, 43.74),
        volume=volume,
        departure_time_s=dep_s,
    )


def test_trip_stream_adapter_batches_and_slices():
    stream = TripStreamAdapter(
        [_trip(7200), _trip(0), _trip(3500), _trip(3700), _trip(10800)],
        sort_by_departure=True,
    )

    batches = list(stream.iter_batches(2))
    assert [len(b.trips) for b in batches] == [2, 2, 1]
    assert batches[0].trips[0].departure_time_s == 0

    slices = list(stream.iter_time_slices(bin_width_s=3600, max_batch_size=2))
    assert [s.departure_bin for s in slices] == [0, 1, 2, 3]
    assert [len(s.trips) for s in slices] == [2, 1, 1, 1]


def test_matrix_assignment_solver_delegates_to_assignment_loop():
    config = AssignmentConfig(max_iterations=2)
    solver = MatrixAssignmentSolver("network.osrm", config)
    demand = ODMatrixAdapter(
        origins=[(1.0, 2.0)],
        destinations=[(3.0, 4.0)],
        matrix=[[5.0]],
    )

    recorded = {}

    class FakeLoop:
        def run(self, trips, *, state_patch=None, progress_callback=None):
            recorded["trips"] = list(trips)
            recorded["state_patch"] = state_patch
            recorded["progress_callback"] = progress_callback
            return "ok"

    solver._make_loop = lambda: FakeLoop()  # type: ignore[method-assign]

    assert solver.solve(demand) == "ok"
    assert len(recorded["trips"]) == 1
    assert recorded["trips"][0].volume == 5.0


def test_matrix_free_solver_exposes_stream_batches():
    config = AssignmentConfig(bin_width_s=3600)
    solver = MatrixFreeHillClimber("network.osrm", config, default_batch_size=2)
    stream = TripStreamAdapter([_trip(0), _trip(1200), _trip(3700), _trip(8000)])

    slices = solver.iter_time_slices(stream, max_batch_size=2)
    assert [s.departure_bin for s in slices] == [0, 1, 2]
    assert [len(s.trips) for s in slices] == [2, 1, 1]


def test_matrix_free_solver_runs_each_batch_independently():
    solver = MatrixFreeHillClimber("network.osrm", AssignmentConfig(), default_batch_size=2)
    stream = TripStreamAdapter([_trip(0), _trip(1), _trip(2)])

    calls = []

    class FakeLoop:
        def run(self, trips, *, state_patch=None, progress_callback=None):
            calls.append(len(list(trips)))
            return types.SimpleNamespace(iterations=1)

    solver._make_loop = lambda: FakeLoop()  # type: ignore[method-assign]

    results = solver.run_independent_batches(stream)
    assert calls == [2, 1]
    assert [r.iterations for r in results] == [1, 1]


def test_assignment_loop_snap_preserves_departure_time():
    loop = AssignmentLoop("network.osrm", AssignmentConfig())

    class FakeEngine:
        def Nearest(self, coordinates, number=1):
            lon, lat = coordinates[0]
            return {"code": "Ok", "waypoints": [{"location": [lon + 0.001, lat + 0.001]}]}

    trips = [_trip(123.0, volume=4.0)]
    snapped = loop._snap_trips(FakeEngine(), trips)
    assert snapped[0].departure_time_s == 123.0
    assert snapped[0].volume == 4.0


def test_matrix_free_solver_runs_statefully_across_slices(monkeypatch):
    solver = MatrixFreeHillClimber("network.osrm", AssignmentConfig(bin_width_s=3600))
    stream = TripStreamAdapter([_trip(0.0), _trip(10.0), _trip(3700.0)])

    state = NetworkState.from_edges(
        from_ids=np.array([1], dtype=np.uint64),
        to_ids=np.array([2], dtype=np.uint64),
        lengths_m=np.array([100.0]),
        freeflow_kmh=np.array([60.0]),
        jam_density=np.array([150.0]),
        n_lanes=np.array([1], dtype=np.uint8),
    )

    route_calls = []
    customize_calls = []

    class FakeLoop:
        def __init__(self):
            self.base_path = "network.osrm"
            self.config = AssignmentConfig(bin_width_s=3600, verbosity="ERROR")
            self.smoother = types.SimpleNamespace(build_adjacency=lambda *args, **kwargs: None)
            self.writer = types.SimpleNamespace(write_from_state=lambda state, only_changed=True: "/tmp/speeds.csv")

        def _create_engine(self):
            return object()

        def _snap_trips(self, engine, trips):
            return list(trips)

        def _route_and_accumulate_with_paths(self, engine, trips, state_obj):
            route_calls.append(len(trips))
            if state_obj.n_edges == 0:
                state_obj.register_edge(1, 2, 100.0, 60.0, 150.0, 1)
            if len(route_calls) == 1:
                return (
                    np.array([2.0]),
                    np.array([20.0]),
                    100.0,
                    [
                        types.SimpleNamespace(
                            trip_index=i,
                            edge_indices=[0],
                            density_contribution=[1.0],
                            duration_s=50.0,
                        )
                        for i in range(len(trips))
                    ],
                )
            return (
                np.array([3.0]),
                np.array([30.0]),
                200.0,
                [
                    types.SimpleNamespace(
                        trip_index=i,
                        edge_indices=[0],
                        density_contribution=[1.5],
                        duration_s=200.0,
                    )
                    for i in range(len(trips))
                ],
            )

        def _update_state(self, state_obj):
            state_obj.speed_kmh = np.maximum(state_obj.freeflow_kmh - state_obj.density_vpkm, 1.0)
            state_obj.flow_vph = state_obj.density_vpkm * state_obj.speed_kmh

    solver._make_loop = lambda: FakeLoop()  # type: ignore[method-assign]
    monkeypatch.setattr("osrm.assignment.solvers.osrm_module.customize", lambda *args, **kwargs: customize_calls.append(args))

    result = solver.run_stream(stream)

    assert isinstance(result, HillClimberResult)
    assert result.n_batches == 2
    assert route_calls == [2, 1]
    assert len(customize_calls) == 2
    assert np.isclose(result.network_state.density_vpkm[0], 5.0)
    assert [b.departure_bin for b in result.batch_results] == [0, 1]
    assert [b.n_trips for b in result.batch_results] == [2, 1]
    assert result.od_ledger is not None
    assert len(result.od_ledger) == 1
    assert result.od_ledger[0].total_volume == 3.0
    assert np.isclose(result.od_ledger[0].assigned_cost_s, 100.0)


def test_matrix_free_solver_runs_msa(monkeypatch):
    solver = MatrixFreeHillClimber("network.osrm", AssignmentConfig(bin_width_s=3600))
    stream = TripStreamAdapter([
        _trip(0.0, volume=1.0),
        _trip(10.0, volume=1.0),
        DemandTrip(
            origin=(7.43, 43.75),
            destination=(7.44, 43.76),
            volume=1.0,
            departure_time_s=20.0,
        ),
    ])

    state = NetworkState.from_edges(
        from_ids=np.array([1, 2], dtype=np.uint64),
        to_ids=np.array([2, 3], dtype=np.uint64),
        lengths_m=np.array([100.0, 120.0]),
        freeflow_kmh=np.array([60.0, 50.0]),
        jam_density=np.array([150.0, 150.0]),
        n_lanes=np.array([1, 1], dtype=np.uint8),
    )

    route_calls = []
    customize_calls = []

    class FakeEngine:
        pass

    class FakeLoop:
        def __init__(self):
            self.base_path = "network.osrm"
            self.config = AssignmentConfig(bin_width_s=3600, verbosity="ERROR")
            self.smoother = types.SimpleNamespace(build_adjacency=lambda *args, **kwargs: None)
            self.writer = types.SimpleNamespace(write_from_state=lambda state, only_changed=True: "/tmp/speeds.csv")

        def _create_engine(self):
            return FakeEngine()

        def _snap_trips(self, engine, trips):
            return list(trips)

        def _route_and_accumulate_with_paths(self, engine, trips, state_obj):
            route_calls.append(len(trips))
            if state_obj.n_edges == 0:
                state_obj.register_edge(1, 2, 100.0, 60.0, 150.0, 1)
                state_obj.register_edge(2, 3, 100.0, 60.0, 150.0, 1)
            paths = [
                types.SimpleNamespace(
                    trip_index=i,
                    edge_indices=[0],
                    density_contribution=[1.0],
                    duration_s=50.0,
                )
                for i in range(len(trips))
            ]
            return (
                np.array([2.0, 1.0]),
                np.array([20.0, 10.0]),
                180.0,
                paths,
            )

        def _update_state(self, state_obj):
            state_obj.speed_kmh = np.maximum(state_obj.freeflow_kmh - state_obj.density_vpkm, 1.0)
            state_obj.flow_vph = state_obj.density_vpkm * state_obj.speed_kmh

        def _compute_relative_gap(self, state_obj, blended_vol, aon_vol):
            link_cost = state_obj.length_m * 3.6 / np.maximum(state_obj.speed_kmh, 1.0)
            num = float(np.sum(blended_vol * link_cost))
            den = float(np.sum(aon_vol * link_cost))
            return num / den - 1.0 if den > 0 else 0.0

    solver._make_loop = lambda: FakeLoop()  # type: ignore[method-assign]
    monkeypatch.setattr(
        "osrm.assignment.solvers.osrm_module.customize",
        lambda *args, **kwargs: customize_calls.append(args),
    )

    result = solver.run_stream(
        stream,
        sample_rate=1.0,
        max_rounds=2,
        gap_threshold=0.0001,
    )

    # Greedy routes 3 trips, then MSA routes all 3 trips each iteration
    assert route_calls[0] == 3
    assert all(n == 3 for n in route_calls[1:])
    # 1 greedy customize + at least 1 MSA customize
    assert len(customize_calls) >= 2
    # MSA results should be populated
    assert len(result.msa_results) >= 1
    assert result.msa_results[0].alpha == 0.5  # first MSA step: 1/(1+1)
    assert result.msa_results[0].state_change_norm >= 0.0
    assert result.od_ledger is not None
    assert len(result.od_ledger) == 2


def test_fw_rejects_large_sampled_networks():
    """FW raises NotImplementedError when trip count exceeds 100k."""
    import pytest

    trips = [_trip(float(i % 100), volume=1.0) for i in range(100_001)]
    solver = MatrixFreeHillClimber("network.osrm", AssignmentConfig(bin_width_s=3600))

    with pytest.raises(NotImplementedError, match="Frank-Wolfe requires full-pass"):
        solver.run_stream(
            trips,
            sample_rate=0.10,
            max_rounds=1,
            method="fw",
        )
