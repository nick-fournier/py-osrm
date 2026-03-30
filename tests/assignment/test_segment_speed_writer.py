"""Tests for SegmentSpeedWriter."""

import numpy as np
import pytest

from osrm.assignment.segment_speed_writer import SegmentSpeedWriter
from osrm.assignment.network_state import NetworkState


@pytest.fixture
def writer(tmp_path):
    return SegmentSpeedWriter(output_dir=str(tmp_path), prefix="test_speeds")


class TestWrite:
    def test_basic_write(self, writer, tmp_path):
        edge_ids = np.array([[100, 200], [300, 400]], dtype=np.uint64)
        speeds = np.array([45.0, 30.0])
        path = writer.write(edge_ids, speeds)

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "100,200,45.0000"
        assert lines[1] == "300,400,30.0000"

    def test_speed_floor(self, writer):
        edge_ids = np.array([[100, 200]], dtype=np.uint64)
        speeds = np.array([0.0])
        path = writer.write(edge_ids, speeds, min_speed_kmh=5.0)

        lines = path.read_text().strip().split("\n")
        assert lines[0] == "100,200,5.0000"

    def test_suffix(self, writer, tmp_path):
        edge_ids = np.array([[100, 200]], dtype=np.uint64)
        speeds = np.array([50.0])
        path = writer.write(edge_ids, speeds, suffix="_period_3")
        assert "test_speeds_period_3.csv" in path.name

    def test_empty_edges(self, writer):
        edge_ids = np.empty((0, 2), dtype=np.uint64)
        speeds = np.empty(0, dtype=np.float64)
        path = writer.write(edge_ids, speeds)
        assert path.exists()
        assert path.read_text() == ""


class TestWriteFromState:
    def test_full_write(self, writer):
        net = NetworkState.from_edges(
            from_ids=np.array([100, 200], dtype=np.uint64),
            to_ids=np.array([200, 300], dtype=np.uint64),
            lengths_m=np.array([1000.0, 500.0]),
            freeflow_kmh=np.array([60.0, 50.0]),
            jam_density=np.array([150.0, 130.0]),
            n_lanes=np.array([2, 1], dtype=np.uint8),
        )
        net.speed_kmh[:] = [45.0, 50.0]

        path = writer.write_from_state(net)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_only_changed(self, writer):
        net = NetworkState.from_edges(
            from_ids=np.array([100, 200], dtype=np.uint64),
            to_ids=np.array([200, 300], dtype=np.uint64),
            lengths_m=np.array([1000.0, 500.0]),
            freeflow_kmh=np.array([60.0, 50.0]),
            jam_density=np.array([150.0, 130.0]),
            n_lanes=np.array([2, 1], dtype=np.uint8),
        )
        # Only edge 0 is changed (60 → 45), edge 1 stays at freeflow
        net.speed_kmh[:] = [45.0, 50.0]

        path = writer.write_from_state(net, only_changed=True, tolerance_kmh=0.5)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert lines[0].startswith("100,200,")


class TestCleanup:
    def test_cleanup_removes_file(self, writer):
        edge_ids = np.array([[100, 200]], dtype=np.uint64)
        speeds = np.array([50.0])
        path = writer.write(edge_ids, speeds)
        assert path.exists()
        writer.cleanup()
        assert not path.exists()

    def test_cleanup_nonexistent_no_error(self, writer):
        writer.cleanup(suffix="_missing")  # Should not raise
