"""End-to-end smoke tests for the multi-period scale test script.

Exercises the full pipeline (build_network → generate_period_csvs →
customize → route) on chi-regional to catch runtime crashes before
the user runs the expensive scale test.  Marked @pytest.mark.slow.
"""

import sys
import shutil
from pathlib import Path

import numpy as np
import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import osrm
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

    Uses 4 periods instead of 96 to keep it fast (~15s total).
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
    def period_csvs(self, network, work_dir):
        """Generate period CSVs (4 periods for speed)."""
        import scale_test_multi_period as smod
        orig = smod.N_PERIODS
        smod.N_PERIODS = 4
        try:
            base, meta = network
            csv_paths, factors, _state = smod.generate_period_csvs(base, meta, work_dir)
        finally:
            smod.N_PERIODS = orig
        return csv_paths, factors

    def test_build_network(self, network):
        """build_network runs without crashing."""
        base, meta = network
        # base is a prefix like "dir/chi_regional.osrm"; actual files have
        # additional suffixes (.ebg, .cell_metrics, etc.)
        osrm_files = list(Path(base).parent.glob(Path(base).name + "*"))
        assert len(osrm_files) > 5, f"Expected OSRM files at {base}, found {osrm_files}"
        assert "zone_centroids" in meta
        assert "od_matrix" in meta
        assert len(meta["zone_centroids"]) > 100

    def test_generate_period_csvs(self, period_csvs):
        """generate_period_csvs runs without crashing."""
        csv_paths, factors = period_csvs
        assert len(csv_paths) == 4
        assert all(Path(p).exists() for p in csv_paths)
        assert factors.shape == (4,)
        assert np.all(factors >= 0)

    def test_customize_and_route(self, network, period_csvs):
        """customize_multi_period + routing with period awareness."""
        from osrm.preprocessing import customize_multi_period
        from osrm.osrm_ext import batch_route_accumulate

        base, meta = network
        csv_paths, _ = period_csvs
        period_dur = 900.0

        period_speed_files = [(p, path) for p, path in enumerate(csv_paths)]
        customize_multi_period(base, period_speed_files=period_speed_files,
                               verbosity="ERROR")

        engine = osrm.OSRM(storage_config=base, algorithm="MLD",
                            use_shared_memory=False)
        centroids = meta["zone_centroids"]
        zone_ids = sorted(centroids.keys())
        rng = np.random.default_rng(42)

        n_pairs = 50
        o_zones = rng.choice(zone_ids, size=n_pairs)
        d_zones = rng.choice(zone_ids, size=n_pairs)

        coords = np.empty((n_pairs, 4), dtype=np.float64)
        valid = 0
        for i in range(n_pairs):
            o, d = int(o_zones[i]), int(d_zones[i])
            if o == d or o not in centroids or d not in centroids:
                continue
            oc, dc = centroids[o], centroids[d]
            coords[valid] = [oc[0], oc[1], dc[0], dc[1]]
            valid += 1
        coords = coords[:valid]
        volumes = np.ones(valid, dtype=np.float64)
        edge_ids = np.zeros((0, 2), dtype=np.uint64)

        vol, tstt, new_edges, _, durations = batch_route_accumulate(
            engine._engine, coords, volumes, edge_ids,
            n_threads=0, return_routes=False,
            departure_period=1, period_duration=period_dur,
            departure_offsets=np.zeros(valid, dtype=np.float64),
            n_periods=0,
        )
        durs = np.asarray(durations)
        routed = np.sum(durs > 0)
        assert routed > 0, "Expected some routes to succeed"

        del engine
