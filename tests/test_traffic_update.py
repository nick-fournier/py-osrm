"""Integration tests for the traffic update surface (Phase 1).

Tests that segment speed and turn penalty CSVs can be passed to
customize() and that the resulting routing reflects the updated weights.
"""

import pytest
import osrm
import tempfile
import shutil
from pathlib import Path

test_dir = Path(__file__).parent
osm_file = test_dir / "data" / "monaco.osm.pbf"
profile_file = test_dir / "data" / "profiles" / "car.lua"


@pytest.fixture(scope="module")
def mld_base(tmp_path_factory):
    """Extract and partition Monaco once for the module."""
    base_dir = tmp_path_factory.mktemp("mld")
    base_path = str(base_dir / "monaco")

    osrm.extract(
        str(osm_file),
        profile="car",
        output_path=base_path,
        verbosity="ERROR",
    )
    osrm.partition(base_path, verbosity="ERROR")
    osrm.customize(base_path, verbosity="ERROR")
    return base_path


def _route_duration(base_path, coords):
    """Route between coords and return total duration in seconds."""
    engine = osrm.OSRM(
        storage_config=base_path,
        algorithm="MLD",
        use_shared_memory=False,
    )
    res = engine.Route(coords)
    return res["routes"][0]["duration"]


class TestCustomizationConfig:
    """Test that the updater_config fields are accessible."""

    def test_segment_speed_paths_default_empty(self):
        config = osrm.CustomizationConfig()
        assert config.segment_speed_lookup_paths == []

    def test_turn_penalty_paths_default_empty(self):
        config = osrm.CustomizationConfig()
        assert config.turn_penalty_lookup_paths == []

    def test_segment_speed_paths_settable(self):
        config = osrm.CustomizationConfig()
        config.segment_speed_lookup_paths = ["/tmp/a.csv", "/tmp/b.csv"]
        assert config.segment_speed_lookup_paths == ["/tmp/a.csv", "/tmp/b.csv"]

    def test_turn_penalty_paths_settable(self):
        config = osrm.CustomizationConfig()
        config.turn_penalty_lookup_paths = ["/tmp/turns.csv"]
        assert config.turn_penalty_lookup_paths == ["/tmp/turns.csv"]


class TestSegmentSpeedCustomize:
    """End-to-end: customize with speed CSV changes routing durations."""

    def test_speed_override_changes_duration(self, mld_base, tmp_path):
        # Pick two coords in Monaco
        coords = [(7.41337, 43.72956), (7.41862, 43.73216)]

        # 1. Get baseline duration
        baseline_dur = _route_duration(mld_base, coords)
        assert baseline_dur > 0

        # 2. Route with annotations to get OSM node IDs on the path
        engine = osrm.OSRM(
            storage_config=mld_base,
            algorithm="MLD",
            use_shared_memory=False,
        )
        route_params = osrm.RouteParameters()
        route_params.coordinates = coords
        route_params.annotations_type = osrm.RouteAnnotationsType.All
        res = engine.Route(route_params)

        nodes = res["routes"][0]["legs"][0]["annotation"]["nodes"]
        assert len(nodes) >= 2, "Route must have at least 2 nodes"

        # 3. Write a speed CSV that slows down ALL segments on this route to 5 km/h
        csv_path = tmp_path / "slow_speeds.csv"
        with open(csv_path, "w") as f:
            for i in range(len(nodes) - 1):
                f.write(f"{int(nodes[i])},{int(nodes[i+1])},5\n")

        # 4. Copy the MLD data to a working directory (avoid mutating shared fixture)
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        work_base = str(work_dir / "monaco")
        for src_file in Path(mld_base).parent.glob("monaco.osrm*"):
            shutil.copy2(src_file, work_dir / src_file.name)

        # 5. Customize with the speed CSV
        result = osrm.customize(
            work_base,
            segment_speed_file=str(csv_path),
            verbosity="ERROR",
        )
        assert result["success"] is True

        # 6. Route again — duration should be significantly longer
        slow_dur = _route_duration(work_base, coords)
        assert slow_dur > baseline_dur * 1.5, (
            f"Expected much slower route with 5 km/h override. "
            f"Baseline: {baseline_dur:.1f}s, Got: {slow_dur:.1f}s"
        )

    def test_speed_override_makes_faster(self, mld_base, tmp_path):
        coords = [(7.41337, 43.72956), (7.41862, 43.73216)]

        # Get baseline
        baseline_dur = _route_duration(mld_base, coords)

        # Route to get nodes
        engine = osrm.OSRM(
            storage_config=mld_base,
            algorithm="MLD",
            use_shared_memory=False,
        )
        route_params = osrm.RouteParameters()
        route_params.coordinates = coords
        route_params.annotations_type = osrm.RouteAnnotationsType.All
        res = engine.Route(route_params)
        nodes = res["routes"][0]["legs"][0]["annotation"]["nodes"]

        # Speed up ALL segments to 200 km/h
        csv_path = tmp_path / "fast_speeds.csv"
        with open(csv_path, "w") as f:
            for i in range(len(nodes) - 1):
                f.write(f"{int(nodes[i])},{int(nodes[i+1])},200\n")

        # Copy and re-customize
        work_dir = tmp_path / "work_fast"
        work_dir.mkdir()
        work_base = str(work_dir / "monaco")
        for src_file in Path(mld_base).parent.glob("monaco.osrm*"):
            shutil.copy2(src_file, work_dir / src_file.name)

        osrm.customize(
            work_base,
            segment_speed_file=str(csv_path),
            verbosity="ERROR",
        )

        fast_dur = _route_duration(work_base, coords)
        assert fast_dur < baseline_dur * 0.8, (
            f"Expected faster route with 200 km/h override. "
            f"Baseline: {baseline_dur:.1f}s, Got: {fast_dur:.1f}s"
        )

    def test_customize_function_accepts_speed_file(self, mld_base, tmp_path):
        """Test that customize() accepts segment_speed_file kwarg."""
        csv_path = tmp_path / "empty.csv"
        csv_path.touch()

        work_dir = tmp_path / "work_empty"
        work_dir.mkdir()
        work_base = str(work_dir / "monaco")
        for src_file in Path(mld_base).parent.glob("monaco.osrm*"):
            shutil.copy2(src_file, work_dir / src_file.name)

        result = osrm.customize(
            work_base,
            segment_speed_file=str(csv_path),
            verbosity="ERROR",
        )
        assert result["success"] is True
