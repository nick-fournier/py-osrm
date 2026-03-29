"""Tests for FractionalLoader."""

import numpy as np
import pytest

from osrm.assignment.fractional_loading import FractionalLoader


@pytest.fixture
def loader():
    return FractionalLoader(bin_width_s=3600.0)  # 1-hour bins


class TestSingleBinTrip:
    def test_all_in_one_bin(self, loader):
        """Short trip fully within one bin."""
        durations = np.array([60.0, 120.0, 180.0])  # 6 min total
        assignments = loader.distribute_route(durations, departure_time_s=0.0)

        # Each link gets fraction=1.0 in bin 0
        assert len(assignments) == 3
        for a in assignments:
            assert a.bin_idx == 0
            assert a.fraction == pytest.approx(1.0)

    def test_fractions_sum_to_one(self, loader):
        """Fractions per link must sum to 1.0."""
        durations = np.array([600.0, 300.0, 900.0])
        assignments = loader.distribute_route(durations, departure_time_s=100.0)

        # Group by link_idx
        by_link: dict[int, float] = {}
        for a in assignments:
            by_link[a.link_idx] = by_link.get(a.link_idx, 0.0) + a.fraction
        for total in by_link.values():
            assert total == pytest.approx(1.0)


class TestMultiBinTrip:
    def test_crosses_one_boundary(self, loader):
        """Trip starting at 58 min, duration 4 min → crosses into bin 1."""
        durations = np.array([240.0])  # 4 min
        dep_time = 58 * 60.0  # 58 min into period
        assignments = loader.distribute_route(durations, dep_time)

        assert len(assignments) == 2
        # First part in bin 0: 2 min / 4 min = 0.5
        assert assignments[0].bin_idx == 0
        assert assignments[0].fraction == pytest.approx(0.5)
        # Second part in bin 1: 2 min / 4 min = 0.5
        assert assignments[1].bin_idx == 1
        assert assignments[1].fraction == pytest.approx(0.5)

    def test_fractions_sum_to_one_multibin(self, loader):
        """45-minute trip across 15-minute loader → multiple bins."""
        loader15 = FractionalLoader(bin_width_s=900.0)  # 15-min bins
        durations = np.array([600.0, 600.0, 600.0, 900.0])  # 45 min
        assignments = loader15.distribute_route(durations, departure_time_s=0.0)

        by_link: dict[int, float] = {}
        for a in assignments:
            by_link[a.link_idx] = by_link.get(a.link_idx, 0.0) + a.fraction
        for total in by_link.values():
            assert total == pytest.approx(1.0)


class TestBoundaryAlignment:
    def test_exact_boundary(self, loader):
        """Trip exactly fills a bin (3600s at t=0) → one assignment."""
        durations = np.array([3600.0])
        assignments = loader.distribute_route(durations, departure_time_s=0.0)
        assert len(assignments) == 1
        assert assignments[0].bin_idx == 0
        assert assignments[0].fraction == pytest.approx(1.0)

    def test_departure_at_boundary(self, loader):
        """Departing exactly at bin boundary."""
        durations = np.array([1800.0])  # 30 min
        dep = 3600.0  # start of bin 1
        assignments = loader.distribute_route(durations, dep)
        assert all(a.bin_idx == 1 for a in assignments)


class TestZeroDuration:
    def test_zero_duration_links_skipped(self, loader):
        durations = np.array([0.0, 300.0, 0.0])
        assignments = loader.distribute_route(durations, departure_time_s=0.0)
        link_indices = {a.link_idx for a in assignments}
        assert 0 not in link_indices
        assert 2 not in link_indices
        assert 1 in link_indices


class TestArrayOutput:
    def test_distribute_to_arrays(self, loader):
        durations = np.array([600.0, 300.0])
        li, bi, frac = loader.distribute_route_to_arrays(durations, 0.0)
        assert len(li) == len(bi) == len(frac)
        assert li.dtype == np.int64
        assert frac.dtype == np.float64


class TestAccumulateRouteFlow:
    def test_accumulate_into_bins(self, loader):
        """Accumulate a route's flow into (n_edges, n_bins) array."""
        edge_ordinals = np.array([5, 10])  # edge ordinals for 2 links
        durations = np.array([1800.0, 1800.0])  # 30 + 30 = 60 min
        flow_bins = np.zeros((20, 3), dtype=np.float64)

        loader.accumulate_route_flow(
            edge_ordinals=edge_ordinals,
            link_durations_s=durations,
            departure_time_s=0.0,
            volume=10.0,  # 10 trips
            flow_bins=flow_bins,
        )
        # Both links in bin 0, volume=10 trips/hr (bin_width=1hr → rate=10)
        assert flow_bins[5, 0] > 0
        assert flow_bins[10, 0] > 0


class TestInvalidInput:
    def test_negative_bin_width_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FractionalLoader(bin_width_s=-1.0)

    def test_zero_bin_width_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FractionalLoader(bin_width_s=0.0)
