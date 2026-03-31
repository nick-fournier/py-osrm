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

        def _discover_network(self, engine, trips):
            return state

        def _route_and_accumulate(self, engine, trips, state_obj):
            route_calls.append(len(trips))
            if len(route_calls) == 1:
                return np.array([2.0]), np.array([20.0]), 100.0
            return np.array([3.0]), np.array([30.0]), 200.0

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
