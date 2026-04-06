"""Integration test: end-to-end assignment on Monaco.

Validates the full assignment pipeline:
  OD matrix → Route → Accumulate → VDF → CSV → Customize → Reload → Converge

Uses a small synthetic OD matrix on the Monaco test network.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

import osrm
from osrm.assignment import (
    AssignmentConfig,
    AssignmentSolver,
    DensitySmoothingConfig,
    ODMatrixAdapter,
    DemandTrip,
)


def _prepare_monaco_network(tmp_path: Path) -> str:
    """Prepare Monaco MLD data in a temp directory."""
    work = tmp_path / "monaco_assignment"
    work.mkdir(parents=True, exist_ok=True)
    src_dir = Path("tests/data")
    pbf = src_dir / "monaco.osm.pbf"
    if not pbf.exists():
        pytest.skip("Monaco PBF not found")

    base = str(work / "monaco")
    osrm.extract(str(pbf), profile="car", output_path=base, verbosity="ERROR")
    osrm.partition(base, verbosity="ERROR")
    osrm.customize(base, verbosity="ERROR")
    return base


def _copy_clean_osrm(base_path: str, run_dir: Path) -> str:
    """Copy clean OSRM files so each run starts from an uncustomized base."""
    src = Path(base_path).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in src.glob("monaco.osrm*"):
        shutil.copy2(f, run_dir / f.name)
    return str(run_dir / Path(base_path).name)


@pytest.fixture(scope="module")
def monaco_mld(tmp_path_factory):
    """Prepare Monaco MLD data in a temp directory."""
    work = tmp_path_factory.mktemp("monaco_assignment")
    return _prepare_monaco_network(work)


@pytest.fixture
def monaco_work(monaco_mld, tmp_path):
    """Copy Monaco MLD data so each test gets a fresh set."""
    work = tmp_path / "work"
    work.mkdir()
    base = str(work / "monaco")
    for f in Path(monaco_mld).parent.glob("monaco.osrm*"):
        shutil.copy2(f, work / f.name)
    return base


def _get_sample_coords(base_path, n=8, seed=42):
    """Get sample routable coordinates from the Monaco network."""
    engine = osrm.OSRM(
        storage_config=base_path, algorithm="MLD", use_shared_memory=False,
    )
    rng = np.random.default_rng(seed)
    # Monaco bounding box (approximate)
    lons = rng.uniform(7.41, 7.44, n * 3)
    lats = rng.uniform(43.72, 43.74, n * 3)

    coords = []
    for lon, lat in zip(lons, lats):
        try:
            r = engine.Nearest(coordinates=[(float(lon), float(lat))])
            if r.get("code") == "Ok" and r.get("waypoints"):
                wp = r["waypoints"][0]
                coords.append(tuple(wp["location"]))
        except Exception:
            pass
        if len(coords) >= n:
            break

    del engine
    if len(coords) < 4:
        pytest.skip("Could not snap enough coordinates to Monaco network")
    return coords


def _build_monaco_trip_stream(base_path: str) -> list[DemandTrip]:
    """Create a tiny time-binned trip stream for assignment MVP tests."""
    coords = _get_sample_coords(base_path, n=8, seed=123)
    departures = [0.0, 0.0, 900.0, 1200.0, 3700.0, 3900.0]
    volumes = [40.0, 25.0, 35.0, 20.0, 30.0, 15.0]
    trips = []
    for i, (dep, vol) in enumerate(zip(departures, volumes)):
        trips.append(DemandTrip(
            origin=coords[i % len(coords)],
            destination=coords[(i + 3) % len(coords)],
            volume=vol,
            departure_time_s=dep,
        ))
    return trips


class TestAssignmentSolver:
    def test_smoke_runs_to_completion(self, monaco_work):
        """Assignment loop completes without error on Monaco."""
        coords = _get_sample_coords(monaco_work, n=6)

        adapter = ODMatrixAdapter.from_uniform_random(
            coordinates=coords,
            total_demand=500.0,
            seed=42,
        )

        config = AssignmentConfig(
            max_iterations=3,
            convergence_gap=0.0,  # Force all iterations to run
            smoothing=DensitySmoothingConfig(method="none"),
            verbosity="ERROR",
        )

        loop = AssignmentSolver(monaco_work, config)
        result = loop.run(adapter.trips())

        assert result.iterations == 3
        assert result.network_state.n_edges > 0
        assert len(result.iteration_log) == 3
        assert all(r.tstt > 0 for r in result.iteration_log)

    def test_flow_is_nonzero(self, monaco_work):
        """After assignment, some edges should have nonzero flow."""
        coords = _get_sample_coords(monaco_work, n=6)

        adapter = ODMatrixAdapter.from_uniform_random(
            coordinates=coords, total_demand=1000.0, seed=99,
        )

        config = AssignmentConfig(
            max_iterations=2,
            smoothing=DensitySmoothingConfig(method="none"),
            verbosity="ERROR",
        )

        loop = AssignmentSolver(monaco_work, config)
        result = loop.run(adapter.trips())

        assert np.sum(result.network_state.flow_vph > 0) > 0

    def test_speeds_decrease_under_load(self, monaco_work):
        """With enough demand, VDF should reduce speeds below freeflow."""
        coords = _get_sample_coords(monaco_work, n=8)

        adapter = ODMatrixAdapter.from_uniform_random(
            coordinates=coords, total_demand=5000.0, seed=77,
        )

        config = AssignmentConfig(
            max_iterations=3,
            smoothing=DensitySmoothingConfig(method="none"),
            verbosity="ERROR",
        )

        loop = AssignmentSolver(monaco_work, config)
        result = loop.run(adapter.trips())

        state = result.network_state
        loaded = state.flow_vph > 0
        if np.any(loaded):
            # At least some loaded edges should have speed < freeflow
            speed_ratio = state.speed_kmh[loaded] / state.freeflow_kmh[loaded]
            assert np.any(speed_ratio < 0.99), "Expected some congestion under heavy load"

    def test_gap_decreases(self, monaco_work):
        """Relative gap should generally decrease across iterations."""
        coords = _get_sample_coords(monaco_work, n=6)

        adapter = ODMatrixAdapter.from_uniform_random(
            coordinates=coords, total_demand=2000.0, seed=55,
        )

        config = AssignmentConfig(
            max_iterations=5,
            convergence_gap=0.001,
            smoothing=DensitySmoothingConfig(method="none"),
            verbosity="ERROR",
        )

        loop = AssignmentSolver(monaco_work, config)
        result = loop.run(adapter.trips())

        gaps = [r.relative_gap for r in result.iteration_log]
        # |gap| should be lower at the end than the start (allowing some noise)
        assert abs(gaps[-1]) <= abs(gaps[0]) + 0.1, (
            f"Gap magnitude should trend downward: {gaps}"
        )

    def test_iteration_log_structure(self, monaco_work):
        """Iteration log should have expected fields."""
        coords = _get_sample_coords(monaco_work, n=4)

        adapter = ODMatrixAdapter.from_uniform_random(
            coordinates=coords, total_demand=200.0,
        )

        config = AssignmentConfig(
            max_iterations=2,
            convergence_gap=0.0,  # Force all iterations
            smoothing=DensitySmoothingConfig(method="none"),
            verbosity="ERROR",
        )

        loop = AssignmentSolver(monaco_work, config)
        result = loop.run(adapter.trips())

        log_dict = result.log_as_dict()
        assert "iteration" in log_dict
        assert "relative_gap" in log_dict
        assert "tstt" in log_dict
        assert "max_flow_delta" in log_dict
        assert len(log_dict["iteration"]) == 2



class TestODMatrixAdapter:
    def test_trip_generation(self):
        origins = [(7.41, 43.73), (7.42, 43.74)]
        dests = [(7.43, 43.72), (7.44, 43.73)]
        matrix = np.array([[100.0, 200.0], [150.0, 0.0]])

        adapter = ODMatrixAdapter(origins, dests, matrix, min_volume=0.1)
        trips = adapter.trips()

        assert len(trips) == 3  # (0,0), (0,1), (1,0) — (1,1) is zero
        assert adapter.total_demand == 450.0
        assert adapter.n_od_pairs == 3

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            ODMatrixAdapter(
                origins=[(0, 0)],
                destinations=[(1, 1), (2, 2)],
                matrix=np.array([[1.0]]),  # 1×1 but expect 1×2
            )

    def test_uniform_random(self):
        coords = [(7.41, 43.73), (7.42, 43.74), (7.43, 43.72)]
        adapter = ODMatrixAdapter.from_uniform_random(coords, total_demand=1000.0)
        assert adapter.total_demand == pytest.approx(1000.0)
        # Diagonal should be zero (no self-trips)
        assert np.all(np.diag(adapter.matrix) == 0)


class TestCrossPeriodFlowAttribution:
    """Verify 2D period-attributed volume from batch_route_accumulate."""

    def test_volume_conservation(self):
        """2D per-period volume must sum to the same total as 1D."""
        from osrm.osrm_ext import batch_route_accumulate

        base = "tests/data/mld/monaco.osrm"
        engine = osrm.OSRM(
            storage_config=base, algorithm="MLD", use_shared_memory=False,
        )

        coords = np.array([
            [7.4131, 43.7278, 7.4272, 43.7395],
            [7.4200, 43.7310, 7.4230, 43.7340],
            [7.4131, 43.7278, 7.4272, 43.7395],
        ], dtype=np.float64)
        volumes = np.array([10.0, 5.0, 8.0], dtype=np.float64)
        edge_ids = np.zeros((0, 2), dtype=np.uint64)
        empty_offsets = np.empty(0, dtype=np.float64)

        # 1D baseline
        vol1d, _, _, _ = batch_route_accumulate(
            engine._engine, coords, volumes, edge_ids,
            n_threads=1, return_routes=False,
            departure_period=-1, period_duration=0.0,
            departure_offsets=empty_offsets, n_periods=0,
        )
        total_1d = float(np.sum(np.asarray(vol1d)))

        # 2D with 900s periods
        offsets = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        vol2d, _, _, _ = batch_route_accumulate(
            engine._engine, coords, volumes, edge_ids,
            n_threads=1, return_routes=False,
            departure_period=0, period_duration=900.0,
            departure_offsets=offsets, n_periods=4,
        )
        v2d = np.asarray(vol2d)
        total_2d = float(np.sum(v2d))

        assert v2d.shape[0] == 4
        assert total_2d == pytest.approx(total_1d, rel=1e-9)

    def test_late_departure_spills_to_next_period(self):
        """A trip departing late in a period should spill flow into the next."""
        from osrm.osrm_ext import batch_route_accumulate

        base = "tests/data/mld/monaco.osrm"
        engine = osrm.OSRM(
            storage_config=base, algorithm="MLD", use_shared_memory=False,
        )

        # Single long-ish route across Monaco
        coords = np.array([
            [7.4131, 43.7278, 7.4272, 43.7395],
        ], dtype=np.float64)
        volumes = np.array([1.0], dtype=np.float64)
        edge_ids = np.zeros((0, 2), dtype=np.uint64)

        # Depart 800s into a 900s period — 100s left before period boundary
        offsets = np.array([800.0], dtype=np.float64)
        vol2d, _, _, _ = batch_route_accumulate(
            engine._engine, coords, volumes, edge_ids,
            n_threads=1, return_routes=False,
            departure_period=0, period_duration=900.0,
            departure_offsets=offsets, n_periods=4,
        )
        v2d = np.asarray(vol2d)

        p0_flow = float(np.sum(v2d[0]))
        p1_flow = float(np.sum(v2d[1]))

        # Both periods should have some flow (trip straddles boundary)
        assert p0_flow > 0, "Period 0 should have flow (early segments)"
        assert p1_flow > 0, "Period 1 should have flow (spill from late departure)"
        # Total should equal trip volume × number of segments
        assert float(np.sum(v2d)) == pytest.approx(p0_flow + p1_flow)

    def test_2d_shape_matches_n_periods(self):
        """Returned array should have shape (n_periods, n_edges)."""
        from osrm.osrm_ext import batch_route_accumulate

        base = "tests/data/mld/monaco.osrm"
        engine = osrm.OSRM(
            storage_config=base, algorithm="MLD", use_shared_memory=False,
        )

        coords = np.array([
            [7.4200, 43.7310, 7.4230, 43.7340],
        ], dtype=np.float64)
        volumes = np.array([1.0], dtype=np.float64)
        edge_ids = np.zeros((0, 2), dtype=np.uint64)
        offsets = np.array([0.0], dtype=np.float64)

        for n_per in [2, 8, 96]:
            vol, _, _, _ = batch_route_accumulate(
                engine._engine, coords, volumes, edge_ids,
                n_threads=1, return_routes=False,
                departure_period=0, period_duration=900.0,
                departure_offsets=offsets, n_periods=n_per,
            )
            v = np.asarray(vol)
            assert v.ndim == 2
            assert v.shape[0] == n_per

    def test_n_periods_zero_returns_1d(self):
        """When n_periods=0, should return 1D array (legacy behavior)."""
        from osrm.osrm_ext import batch_route_accumulate

        base = "tests/data/mld/monaco.osrm"
        engine = osrm.OSRM(
            storage_config=base, algorithm="MLD", use_shared_memory=False,
        )

        coords = np.array([
            [7.4200, 43.7310, 7.4230, 43.7340],
        ], dtype=np.float64)
        volumes = np.array([1.0], dtype=np.float64)
        edge_ids = np.zeros((0, 2), dtype=np.uint64)

        vol, _, _, _ = batch_route_accumulate(
            engine._engine, coords, volumes, edge_ids,
            n_threads=1, return_routes=False,
            departure_period=-1, period_duration=0.0,
            departure_offsets=np.empty(0, dtype=np.float64),
            n_periods=0,
        )
        assert np.asarray(vol).ndim == 1
