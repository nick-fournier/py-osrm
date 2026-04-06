"""End-to-end smoke tests for the multi-period scale test script.

Exercises the full pipeline (build_network → build_demand →
run_assignment → generate_report) on chi-regional to catch runtime
crashes before the user runs the expensive scale test.
Marked @pytest.mark.slow.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from osrm.assignment.osm_synthesis import LinkClass, classify_by_speed
from osrm.assignment.tntp import TNTPLink, parse_net

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "chicago_regional"


def _make_link(**kwargs):
    """Create a TNTPLink with sensible defaults."""
    defaults = dict(
        init_node=1, term_node=2, capacity=1800, length=1.0,
        free_flow_time=1.0, b=0.15, power=4.0, speed=25.0,
        toll=0.0, link_type=1,
    )
    defaults.update(kwargs)
    return TNTPLink(**defaults)


def _get_override():
    """Import the override from the scale test script."""
    from scale_test_multi_period import _regional_classify_override
    return _regional_classify_override


# ── unit tests for the override function ────────────────────────────


class TestRegionalClassifyOverride:
    """Validate _regional_classify_override uses correct LinkClass attrs."""

    def test_returns_none_for_normal_link(self):
        override = _get_override()
        link = _make_link(link_type=1)
        default_cls = LinkClass(highway="secondary", n_lanes=2, speed_kmh=60.0)
        assert override(link, 500.0, default_cls) is None

    def test_link_type_3_becomes_motorway_link(self):
        override = _get_override()
        link = _make_link(link_type=3)
        default_cls = LinkClass(highway="tertiary", n_lanes=1, speed_kmh=40.0)
        result = override(link, 500.0, default_cls)
        assert result is not None
        assert result.highway == "motorway_link"
        assert result.n_lanes == 2  # max(2, 1) = 2
        assert result.speed_kmh == 40.0  # min(40, 100) = 40

    def test_link_type_3_preserves_high_lane_count(self):
        override = _get_override()
        link = _make_link(link_type=3)
        default_cls = LinkClass(highway="primary", n_lanes=4, speed_kmh=80.0)
        result = override(link, 500.0, default_cls)
        assert result.n_lanes == 4  # max(2, 4) = 4

    def test_link_type_3_caps_speed_at_100(self):
        override = _get_override()
        link = _make_link(link_type=3)
        default_cls = LinkClass(highway="primary", n_lanes=3, speed_kmh=120.0)
        result = override(link, 500.0, default_cls)
        assert result.speed_kmh == 100.0

    def test_speed_over_130_gets_capped(self):
        override = _get_override()
        link = _make_link(link_type=1)
        default_cls = LinkClass(highway="motorway", n_lanes=3, speed_kmh=150.0)
        result = override(link, 500.0, default_cls)
        assert result is not None
        assert result.speed_kmh == 130.0
        assert result.n_lanes == 3

    def test_speed_exactly_130_no_override(self):
        override = _get_override()
        link = _make_link(link_type=1)
        default_cls = LinkClass(highway="motorway", n_lanes=3, speed_kmh=130.0)
        assert override(link, 500.0, default_cls) is None


# ── TNTP parsing sanity ─────────────────────────────────────────────


class TestChiRegionalParsing:
    """Verify TNTP data parses and classifies without errors."""

    @pytest.fixture(scope="class")
    def net(self):
        return parse_net(FIXTURE_DIR / "ChicagoRegional_net.tntp")

    def test_network_parses(self, net):
        assert len(net.links) == 39018

    def test_all_links_have_required_attrs(self, net):
        for link in net.links[:100]:
            assert hasattr(link, "link_type")
            assert hasattr(link, "capacity")
            assert hasattr(link, "speed")
            assert not hasattr(link, "lanes"), "TNTPLink should not have 'lanes'"

    def test_override_applied_to_real_links(self, net):
        """Run the override on every chi-regional link to catch field mismatches."""
        override = _get_override()
        overridden = 0
        for link in net.links:
            speed_kmh = link.speed * 1.60934 if link.speed < 200 else link.speed
            default_cls = classify_by_speed(max(speed_kmh, 1.0), link.capacity)
            result = override(link, 100.0, default_cls)
            if result is not None:
                overridden += 1
                assert hasattr(result, "n_lanes")
                assert hasattr(result, "speed_kmh")
        assert overridden > 0, "Expected at least some links to be overridden"


# ── end-to-end pipeline smoke ───────────────────────────────────────


@pytest.mark.slow
class TestScalePipelineSmoke:
    """Run the actual scale test pipeline functions to catch runtime errors.

    Uses tiny demand_scale (0.001) to keep the assignment fast.
    """

    @pytest.fixture(scope="class")
    def work_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("scale_smoke")

    @pytest.fixture(scope="class")
    def network(self, work_dir):
        """Build network (extract + partition + customize)."""
        from scale_test_multi_period import build_network
        base, meta = build_network(work_dir)
        return base, meta

    @pytest.fixture(scope="class")
    def trips(self, network):
        """Build demand with tiny scale."""
        import scale_test_multi_period as smod
        _, meta = network
        # Use 4 periods to keep it fast
        orig = smod.N_PERIODS
        smod.N_PERIODS = 4
        try:
            trips = smod.build_demand(meta, demand_scale=0.001)
        finally:
            smod.N_PERIODS = orig
        return trips

    def test_build_network(self, network):
        """build_network runs without crashing."""
        base, meta = network
        osrm_files = list(Path(base).parent.glob(Path(base).name + "*"))
        assert len(osrm_files) > 5, f"Expected OSRM files at {base}, found {osrm_files}"
        assert "zone_centroids" in meta
        assert "od_matrix" in meta
        assert len(meta["zone_centroids"]) > 100

    def test_build_demand(self, trips):
        """build_demand produces valid DemandTrip objects."""
        assert len(trips) > 0, "Expected at least some trips"
        for t in trips[:10]:
            assert hasattr(t, "origin")
            assert hasattr(t, "destination")
            assert hasattr(t, "volume")
            assert hasattr(t, "departure_time_s")
            assert t.volume > 0
            assert t.departure_time_s >= 0

    def test_run_assignment(self, network, trips, work_dir):
        """assign_stream runs without crashing on tiny demand."""
        import scale_test_multi_period as smod
        base, meta = network
        # Use 4 periods
        orig = smod.N_PERIODS
        smod.N_PERIODS = 4
        try:
            result = smod.run_assignment(base, meta, trips, work_dir)
        finally:
            smod.N_PERIODS = orig
        assert result.n_batches >= 1
        assert result.n_trips >= 1
        assert result.network_state is not None
        assert result.batch_log is not None
        assert len(result.batch_log) >= 1
