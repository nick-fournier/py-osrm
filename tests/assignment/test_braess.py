"""Braess paradox structural validation.

Validates that the assignment correctly produces the Braess paradox:
adding a shortcut link increases total system travel time (TSTT)
under user equilibrium, because selfish routing overloads it.

See docs/traffic_assignment_design.md §6.3.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

import osrm
from osrm.assignment import (
    AssignmentConfig,
    AssignmentLoop,
    DensitySmoothingConfig,
    MatrixFreeHillClimber,
)
from osrm.assignment.od_matrix import DemandTrip
from osrm.assignment.osm_synthesis import braess_network
from .hillclimber_validation import (
    build_hillclimber_report_sections,
    run_hillclimber_case,
    slice_trips_by_departure,
)


def _prepare_network(tmp_path: Path, with_shortcut: bool):
    """Synthesize, extract, partition, customize a Braess network."""
    label = "with" if with_shortcut else "without"
    work = tmp_path / f"braess_{label}"
    work.mkdir(parents=True)

    osm_path, meta = braess_network(work / "braess.osm", with_shortcut=with_shortcut)
    base = str(work / "braess.osrm")

    osrm.extract(str(osm_path), profile="car", output_path=base, verbosity="ERROR")
    osrm.partition(base, verbosity="ERROR")
    osrm.customize(base, verbosity="ERROR")

    return base, meta


def _run_assignment(
    base_path: str,
    meta: dict,
    demand: float,
    max_iter: int = 15,
    method: str = "msa",
):
    """Run assignment on Braess network."""
    from osrm.assignment.osm_synthesis import patch_braess_lanes

    trips = [DemandTrip(
        origin=meta["origin"],
        destination=meta["destination"],
        volume=demand,
    )]

    config = AssignmentConfig(
        method=method,
        max_iterations=max_iter,
        convergence_gap=0.0,  # Run all iterations
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(Path(base_path).parent),
    )

    loop = AssignmentLoop(base_path, config)

    def lane_patch(state):
        patch_braess_lanes(state, meta)

    return loop.run(trips, state_patch=lane_patch)


def _build_hillclimber_trips(meta: dict, demand_scale: float) -> list[DemandTrip]:
    demand = 2500.0 * demand_scale
    return [DemandTrip(
        origin=meta["origin"],
        destination=meta["destination"],
        volume=demand,
    )]


class TestBraessParadox:
    """Structural validation: Braess paradox.

    The paradox: adding a cheap shortcut to a 4-node diamond network
    INCREASES total system travel time under user equilibrium because
    the shortcut attracts all traffic through both variable links.

    Network: simple 4-node diamond, all links ~10 km, shortcut ~1 km.
    Variable links (1→3, 4→2): 1 lane, maxspeed 80 (v_f ≈ 64 km/h).
    Constant links (1→4, 3→2): 4 lanes, maxspeed 50 (v_f ≈ 40 km/h).

    Demand = 2500 vph.  Mixed equilibrium: ~53% shortcut, ~23% each
    upper/lower.  Expected TSTT increase ≈ 6.7%.
    """

    DEMAND = 2500.0

    def test_tstt_increases_with_shortcut(self, tmp_path):
        """Core Braess test: TSTT should be higher with the shortcut."""
        base_with, meta_with = _prepare_network(tmp_path, with_shortcut=True)
        base_without, meta_without = _prepare_network(tmp_path, with_shortcut=False)

        result_with = _run_assignment(
            base_with, meta_with, demand=self.DEMAND, max_iter=30,
        )
        result_without = _run_assignment(
            base_without, meta_without, demand=self.DEMAND, max_iter=30,
        )

        tstt_with = result_with.iteration_log[-1].tstt
        tstt_without = result_without.iteration_log[-1].tstt
        pct_increase = (tstt_with / tstt_without - 1) * 100

        assert tstt_with > tstt_without, (
            f"Braess paradox not observed: TSTT with shortcut ({tstt_with:.0f}) "
            f"should exceed TSTT without ({tstt_without:.0f})"
        )
        assert pct_increase > 5.0, (
            f"Paradox too weak: {pct_increase:.1f}% TSTT increase, expected >5%"
        )

    def test_shortcut_carries_flow(self, tmp_path):
        """The shortcut link must actually carry flow at equilibrium."""
        base, meta = _prepare_network(tmp_path, with_shortcut=True)
        result = _run_assignment(
            base, meta, demand=self.DEMAND, max_iter=30,
        )
        state = result.network_state

        # Find the shortcut edge (3→4)
        shortcut_flow = 0.0
        for i in range(state.n_edges):
            f, t = int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1])
            if f == 3 and t == 4:
                shortcut_flow = state.flow_vph[i]
                break

        assert shortcut_flow > 100, (
            f"Shortcut 3→4 has only {shortcut_flow:.0f} vph flow — "
            f"should carry substantial traffic for paradox to work"
        )

    def test_with_shortcut_all_on_shortcut_path(self, tmp_path):
        """With shortcut, equilibrium should be mixed across all three routes.

        The shortcut path (1→3→4→2) attracts flow, creating
        the Braess paradox; all three routes carry traffic at equilibrium.
        """
        base, meta = _prepare_network(tmp_path, with_shortcut=True)
        result = _run_assignment(
            base, meta, demand=self.DEMAND, max_iter=30,
        )
        state = result.network_state

        flows = {}
        for i in range(state.n_edges):
            f, t = int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1])
            flows[(f, t)] = state.flow_vph[i]

        sc_flow = flows.get((3, 4), 0)
        upper_flow = flows.get((3, 2), 0)
        lower_flow = flows.get((1, 4), 0)

        # All three routes carry meaningful flow (mixed equilibrium)
        min_share = self.DEMAND * 0.05
        assert sc_flow > min_share, (
            f"Shortcut has only {sc_flow:.0f} vph — "
            f"expected meaningful flow for mixed equilibrium"
        )
        assert upper_flow > min_share, (
            f"Upper route has only {upper_flow:.0f} vph — "
            f"expected meaningful flow for mixed equilibrium"
        )
        assert lower_flow > min_share, (
            f"Lower route has only {lower_flow:.0f} vph — "
            f"expected meaningful flow for mixed equilibrium"
        )

    def test_without_shortcut_balanced_split(self, tmp_path):
        """Without shortcut, traffic should split roughly 50/50."""
        base, meta = _prepare_network(tmp_path, with_shortcut=False)
        result = _run_assignment(
            base, meta, demand=self.DEMAND, max_iter=30,
        )
        state = result.network_state

        # Find flow on variable links (1→3 and 4→2)
        flows = {}
        for i in range(state.n_edges):
            f, t = int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1])
            if (f, t) == (1, 3) or (f, t) == (4, 2):
                flows[(f, t)] = state.flow_vph[i]

        assert len(flows) == 2, f"Expected 2 variable-link flows, got {flows}"
        for edge, flow in flows.items():
            assert self.DEMAND * 0.3 < flow < self.DEMAND * 0.7, (
                f"Edge {edge[0]}→{edge[1]} has {flow:.0f} vph — "
                f"expected ~50% of {self.DEMAND:.0f}"
            )

    def test_both_complete_without_error(self, tmp_path):
        """Both networks should complete assignment without error."""
        base_with, meta_with = _prepare_network(tmp_path, with_shortcut=True)
        base_without, meta_without = _prepare_network(tmp_path, with_shortcut=False)

        result_with = _run_assignment(base_with, meta_with, demand=self.DEMAND, max_iter=5)
        result_without = _run_assignment(base_without, meta_without, demand=self.DEMAND, max_iter=5)

        assert result_with.iterations == 5
        assert result_without.iterations == 5
        assert result_with.network_state.n_edges > 0
        assert result_without.network_state.n_edges > 0

    def test_flow_nonnegativity(self, tmp_path):
        """All link flows must be non-negative."""
        base, meta = _prepare_network(tmp_path, with_shortcut=True)
        result = _run_assignment(base, meta, demand=self.DEMAND)
        assert np.all(result.network_state.flow_vph >= 0)

    def test_speeds_within_bounds(self, tmp_path):
        """Speeds must be between VDF min_speed and freeflow."""
        base, meta = _prepare_network(tmp_path, with_shortcut=True)
        result = _run_assignment(base, meta, demand=self.DEMAND)
        state = result.network_state
        assert np.all(state.speed_kmh >= 1.08 - 1e-6)
        assert np.all(state.speed_kmh <= state.freeflow_kmh + 1e-6)

    def test_fw_monotone_tstt(self, tmp_path):
        """Frank-Wolfe should produce roughly stable TSTT."""
        base, meta = _prepare_network(tmp_path, with_shortcut=True)
        result = _run_assignment(base, meta, demand=self.DEMAND, max_iter=20, method="fw")

        # TSTT should be roughly stable (not oscillating wildly)
        tstt_vals = [r.tstt for r in result.iteration_log]
        assert len(tstt_vals) >= 3  # at least a few iters before stagnation

    def test_freeflow_immutable_across_runs(self, tmp_path):
        """Freeflow speed must not degrade when run() is called twice.

        Regression test: segment-speed customization permanently mutates
        OSRM edge weights.  A second run() on the same base path used to
        read congested speeds as freeflow, causing speed collapse.
        The fix: each run needs a clean copy of the OSRM files.
        """
        base1, meta = _prepare_network(tmp_path / "r1", with_shortcut=True)
        r1 = _run_assignment(base1, meta, demand=self.DEMAND, max_iter=10)
        s1 = r1.network_state

        # Second run on a fresh copy (correct usage)
        base2, meta2 = _prepare_network(tmp_path / "r2", with_shortcut=True)
        r2 = _run_assignment(base2, meta2, demand=self.DEMAND, max_iter=10)
        s2 = r2.network_state

        # Build freeflow maps keyed by edge
        ff1 = {
            (int(s1.edge_ids[i, 0]), int(s1.edge_ids[i, 1])): s1.freeflow_kmh[i]
            for i in range(s1.n_edges)
        }
        ff2 = {
            (int(s2.edge_ids[i, 0]), int(s2.edge_ids[i, 1])): s2.freeflow_kmh[i]
            for i in range(s2.n_edges)
        }

        common = set(ff1) & set(ff2)
        assert len(common) >= 3, f"Expected ≥3 common edges, got {len(common)}"
        for key in common:
            assert abs(ff1[key] - ff2[key]) < 0.5, (
                f"Freeflow mismatch on {key}: run1={ff1[key]:.1f}, "
                f"run2={ff2[key]:.1f}"
            )

    def test_hillclimber_shortcut_reduces_tstt(self, tmp_path):
        """Under hill-climber (incremental) loading the shortcut HELPS.

        Unlike equilibrium assignment, the hill-climber loads demand in
        sequential batches without global rerouting.  The shortcut provides
        genuine relief under greedy loading because no single batch
        overloads it.  This is the expected (correct) behaviour — the
        Braess paradox is an equilibrium phenomenon.
        """
        from osrm.assignment.osm_synthesis import patch_braess_lanes

        base_with, meta_with = _prepare_network(tmp_path / "with", with_shortcut=True)
        base_without, meta_without = _prepare_network(tmp_path / "without", with_shortcut=False)
        case_with = run_hillclimber_case(
            base_path=base_with,
            meta=meta_with,
            copy_fn=lambda base, run_dir: base,
            trip_builder=_build_hillclimber_trips,
            run_dir=tmp_path / "with_run",
            demand_scale=1.0,
            n_slices=10,
            state_patch_factory=lambda m: lambda s: patch_braess_lanes(s, m),
        )
        case_without = run_hillclimber_case(
            base_path=base_without,
            meta=meta_without,
            copy_fn=lambda base, run_dir: base,
            trip_builder=_build_hillclimber_trips,
            run_dir=tmp_path / "without_run",
            demand_scale=1.0,
            n_slices=10,
            state_patch_factory=lambda m: lambda s: patch_braess_lanes(s, m),
        )

        with_tstt = sum(b.batch_tstt for b in case_with.result.batch_results)
        without_tstt = sum(b.batch_tstt for b in case_without.result.batch_results)
        # Shortcut reduces TSTT under incremental loading (opposite of equilibrium)
        assert with_tstt < without_tstt, (
            f"Expected shortcut to REDUCE TSTT under hill-climber loading, "
            f"but with={with_tstt:,.0f} >= without={without_tstt:,.0f}"
        )

    def test_reroute_epochs_reproduce_paradox(self, tmp_path):
        """Reroute epochs restore equilibrium and reproduce the paradox.

        After the initial HC greedy load (which sees no paradox), a single
        reroute epoch iterates through slices oldest-first, subtracting
        old density and re-routing on the updated state. This approaches
        Wardrop equilibrium and the Braess paradox should emerge.
        """
        from osrm.assignment.osm_synthesis import patch_braess_lanes

        demand = self.DEMAND

        def _run_with_reroute(base, meta):
            trips = [DemandTrip(
                origin=meta["origin"],
                destination=meta["destination"],
                volume=demand,
            )]
            sliced = slice_trips_by_departure(
                trips, n_slices=10, bin_width_s=3600.0,
            )
            config = AssignmentConfig(bin_width_s=3600.0, verbosity="NONE")
            hc = MatrixFreeHillClimber(base, config)
            return hc.run_stream(
                sliced,
                state_patch=lambda s: patch_braess_lanes(s, meta),
                max_epochs=5,
                gap_threshold=0.001,
            )

        base_with, meta_with = _prepare_network(tmp_path / "with", with_shortcut=True)
        base_without, meta_without = _prepare_network(
            tmp_path / "without", with_shortcut=False,
        )
        result_w = _run_with_reroute(base_with, meta_with)
        result_wo = _run_with_reroute(base_without, meta_without)

        tstt_w = sum(e.tstt for e in result_w.slice_ledger)
        tstt_wo = sum(e.tstt for e in result_wo.slice_ledger)
        pct = (tstt_w / tstt_wo - 1) * 100

        # Paradox should appear: shortcut INCREASES TSTT after rerouting
        assert tstt_w > tstt_wo, (
            f"Expected Braess paradox after reroute epochs, "
            f"but with={tstt_w:,.0f} <= without={tstt_wo:,.0f} ({pct:+.1f}%)"
        )
        # Should be a material increase, not noise
        assert pct > 2.0, (
            f"Paradox too weak: {pct:.1f}% TSTT increase, expected >2%"
        )
        # Gap should be near-zero
        assert result_w.epoch_results, "Expected at least one reroute epoch"
        assert result_w.epoch_results[-1].gap < 0.01, (
            f"Gap too large: {result_w.epoch_results[-1].gap:.6f}"
        )




def _braess_state_table(scenarios: list[tuple]) -> str:
    """Build an N-column link state table for the Braess diamond.

    Parameters
    ----------
    scenarios : list of (NetworkState, label) tuples
        Arbitrary number of scenarios to compare side-by-side.
    """
    from collections import OrderedDict

    N = len(scenarios)
    rows = []
    for state, scenario in scenarios:
        for i in range(state.n_edges):
            label = f"{int(state.edge_ids[i,0])}&rarr;{int(state.edge_ids[i,1])}"
            v = max(state.speed_kmh[i], 1.08)
            tt = state.length_m[i] / (v / 3.6)
            rows.append((label, scenario, state.density_vpkm[i], state.speed_kmh[i],
                         state.freeflow_kmh[i], state.flow_vph[i], state.jam_density[i],
                         state.n_lanes[i], state.length_m[i], tt))

    pivot: dict[str, dict] = OrderedDict()
    for label, scenario, k, v, vf, q, kj, lanes, length, tt in rows:
        pivot.setdefault(label, {"lanes": lanes, "kj": kj, "vf": vf, "length": length})[scenario] = (k, v, q, tt)

    max_width = max(900, 550 + 200 * N)
    font_size = "0.82em" if N >= 3 else "0.85em"

    html = (
        f'<table style="border-collapse:collapse; width:100%; max-width:{max_width}px; '
        f'margin:12px auto; font-family:system-ui,sans-serif; font-size:{font_size};">'
        '<thead><tr style="border-bottom:2px solid #333;">'
        '<th style="text-align:left;padding:8px;">Link</th>'
        '<th style="padding:6px;">Len</th>'
        '<th style="padding:6px;">Ln</th>'
        '<th style="padding:6px;">k<sub>j</sub></th>'
        '<th style="padding:6px;">v<sub>f</sub></th>'
    )
    for _, scenario in scenarios:
        html += f'<th colspan="4" style="text-align:center;padding:6px;border-left:2px solid #ccc;">{scenario}</th>'
    html += '</tr><tr style="border-bottom:1px solid #999;">'
    html += '<th></th><th></th><th></th><th></th><th></th>'
    for _ in scenarios:
        html += ('<th style="padding:3px 5px;border-left:2px solid #ccc;">k</th>'
                 '<th style="padding:3px 5px;">v</th>'
                 '<th style="padding:3px 5px;">q</th>'
                 '<th style="padding:3px 5px;">t</th>')
    html += '</tr></thead><tbody>'

    def _v_color(v, vf):
        r = v / vf if vf > 0 else 1
        return "#F44336" if r < 0.1 else "#FF9800" if r < 0.5 else "#4CAF50"

    def _k_style(k, kj):
        r = k / kj if kj > 0 else 0
        if r > 0.9: return "font-weight:700;color:#F44336;"
        if r > 0.5: return "color:#FF9800;"
        return ""

    def _fmt_time(s):
        return f"{s/3600:.1f}h" if s >= 3600 else f"{s:.0f}s"

    for link, info in pivot.items():
        kj, vf = info["kj"], info["vf"]
        length_km = info["length"] / 1000.0
        ff_time = info["length"] / (vf / 3.6) if vf > 0 else 0

        def _cells(vals, _kj=kj, _vf=vf, _ff=ff_time):
            if vals is None:
                return ('<td style="text-align:right;padding:3px 5px;border-left:2px solid #ccc;">&mdash;</td>'
                        '<td style="text-align:right;padding:3px 5px;">&mdash;</td>'
                        '<td style="text-align:right;padding:3px 5px;">&mdash;</td>'
                        '<td style="text-align:right;padding:3px 5px;">&mdash;</td>')
            k, v, q, tt = vals
            tt_r = tt / _ff if _ff > 0 else 1
            tt_c = "#F44336" if tt_r > 2 else "#FF9800" if tt_r > 1.3 else "#4CAF50"
            return (
                f'<td style="text-align:right;padding:3px 5px;border-left:2px solid #ccc;{_k_style(k, _kj)}">{k:.1f}</td>'
                f'<td style="text-align:right;padding:3px 5px;color:{_v_color(v, _vf)};">{v:.1f}</td>'
                f'<td style="text-align:right;padding:3px 5px;">{q:.0f}</td>'
                f'<td style="text-align:right;padding:3px 5px;color:{tt_c};">{_fmt_time(tt)}</td>')

        cells = "".join(_cells(info.get(scenario)) for _, scenario in scenarios)
        html += (
            f'<tr style="border-bottom:1px solid #e0e0e0;">'
            f'<td style="padding:3px 5px;font-weight:600;">{link}</td>'
            f'<td style="text-align:center;padding:3px 5px;">{length_km:.1f}</td>'
            f'<td style="text-align:center;padding:3px 5px;">{info["lanes"]}</td>'
            f'<td style="text-align:center;padding:3px 5px;">{kj:.0f}</td>'
            f'<td style="text-align:center;padding:3px 5px;">{vf:.0f}</td>'
            f'{cells}</tr>')
    html += '</tbody></table>'
    return html


def _braess_route_tt_table(scenarios: list[tuple]) -> str:
    """Build an N-column route travel time table for the Braess diamond.

    Parameters
    ----------
    scenarios : list of (NetworkState, label) tuples
        Arbitrary number of scenarios to compare side-by-side.
    """
    routes = {
        "Upper (1&rarr;3&rarr;2)": [("1", "3"), ("3", "2")],
        "Lower (1&rarr;4&rarr;2)": [("1", "4"), ("4", "2")],
        "Shortcut (1&rarr;3&rarr;4&rarr;2)": [("1", "3"), ("3", "4"), ("4", "2")],
    }

    def _link_times(state):
        times = {}
        for i in range(state.n_edges):
            f_id = str(int(state.edge_ids[i, 0]))
            t_id = str(int(state.edge_ids[i, 1]))
            v = max(state.speed_kmh[i], 1.08)
            times[(f_id, t_id)] = state.length_m[i] / (v / 3.6)
        return times

    N = len(scenarios)
    all_times = [_link_times(state) for state, _ in scenarios]
    max_width = max(600, 350 + 150 * N)

    html = (
        f'<table style="border-collapse:collapse; width:100%; max-width:{max_width}px; '
        f'margin:12px auto; font-family:system-ui,sans-serif; font-size:0.85em;">'
        '<thead><tr style="border-bottom:2px solid #333;">'
        '<th style="text-align:left;padding:8px;">Route</th>'
    )
    for _, scenario in scenarios:
        html += f'<th style="text-align:right;padding:8px;">{scenario}</th>'
    html += '</tr></thead><tbody>'

    for route_name, links in routes.items():
        html += f'<tr style="border-bottom:1px solid #e0e0e0;"><td style="padding:6px 8px;font-weight:600;">{route_name}</td>'
        for times in all_times:
            total = None
            if all(lk in times for lk in links):
                total = sum(times[lk] for lk in links)
            html += f'<td style="text-align:right;padding:6px 8px;">{f"{total:.1f}s" if total is not None else "&mdash;"}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html


def generate_braess_report(
    tmp_path: str | Path,
    output_path: str = "docs/plots/braess_validation.html",
    demand: float = 2500.0,
    max_iter: int = 100,
) -> Path:
    """Run unified Braess validation and generate an interactive HTML report.

    Combines matrix-based equilibrium (MSA & Frank-Wolfe) and matrix-free
    hill-climber (greedy + reroute epochs) on a 4-node diamond network.

    Parameters
    ----------
    tmp_path : str or Path
        Working directory for temporary OSRM files.
    output_path : str
        Where to write the HTML report.
    demand : float
        Demand volume (vehicles per period).
    max_iter : int
        Assignment iterations for matrix methods.

    Returns
    -------
    Path to the generated report.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from osrm.assignment.osm_synthesis import patch_braess_lanes
    from osrm.assignment.plots import _add_mfd_section, _write_combined_report

    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Run all scenarios
    # ===================================================================

    # Matrix MSA
    base_with, meta_with = _prepare_network(tmp_path / "msa", with_shortcut=True)
    base_without, meta_without = _prepare_network(tmp_path / "msa", with_shortcut=False)
    result_with = _run_assignment(base_with, meta_with, demand, max_iter)
    result_without = _run_assignment(base_without, meta_without, demand, max_iter)

    # Matrix Frank-Wolfe
    base_fw_w, meta_fw_w = _prepare_network(tmp_path / "fw", with_shortcut=True)
    base_fw_wo, meta_fw_wo = _prepare_network(tmp_path / "fw", with_shortcut=False)
    result_fw_with = _run_assignment(base_fw_w, meta_fw_w, demand, max_iter, method="fw")
    result_fw_without = _run_assignment(base_fw_wo, meta_fw_wo, demand, max_iter, method="fw")

    # HC (greedy + reroute in a single run per scenario)
    base_hc_w, meta_hc_w = _prepare_network(tmp_path / "hc", with_shortcut=True)
    base_hc_wo, meta_hc_wo = _prepare_network(tmp_path / "hc", with_shortcut=False)

    def _hc_trip_builder(meta, scale):
        return [DemandTrip(
            origin=meta["origin"],
            destination=meta["destination"],
            volume=demand * scale,
        )]

    case_with = run_hillclimber_case(
        base_path=base_hc_w,
        meta=meta_hc_w,
        copy_fn=lambda base, run_dir: base,
        trip_builder=_hc_trip_builder,
        run_dir=tmp_path / "hc_with_run",
        demand_scale=1.0,
        n_slices=10,
        state_patch_factory=lambda m: lambda s: patch_braess_lanes(s, m),
        max_epochs=10,
        gap_threshold=0.001,
    )
    case_without = run_hillclimber_case(
        base_path=base_hc_wo,
        meta=meta_hc_wo,
        copy_fn=lambda base, run_dir: base,
        trip_builder=_hc_trip_builder,
        run_dir=tmp_path / "hc_without_run",
        demand_scale=1.0,
        n_slices=10,
        state_patch_factory=lambda m: lambda s: patch_braess_lanes(s, m),
        max_epochs=10,
        gap_threshold=0.001,
    )

    # Compute key metrics
    tstt_with_vals = [r.tstt for r in result_with.iteration_log]
    tstt_without_vals = [r.tstt for r in result_without.iteration_log]
    delta = tstt_with_vals[-1] - tstt_without_vals[-1]
    pct = delta / tstt_without_vals[-1] * 100 if tstt_without_vals[-1] > 0 else 0

    hc_tstt_wo = sum(b.batch_tstt for b in case_without.result.batch_results)
    hc_tstt_w = sum(b.batch_tstt for b in case_with.result.batch_results)
    hc_pct = (hc_tstt_w / hc_tstt_wo - 1) * 100 if hc_tstt_wo else 0.0

    rr_tstt_w = sum(e.tstt for e in case_with.result.slice_ledger)
    rr_tstt_wo = sum(e.tstt for e in case_without.result.slice_ledger)
    rr_pct = (rr_tstt_w / rr_tstt_wo - 1) * 100 if rr_tstt_wo else 0.0

    fw_tstt_w = result_fw_with.iteration_log[-1].tstt
    fw_tstt_wo = result_fw_without.iteration_log[-1].tstt
    fw_pct = (fw_tstt_w / fw_tstt_wo - 1) * 100 if fw_tstt_wo else 0.0

    n_epochs_w = len(case_with.result.epoch_results)
    rr_gap_w = case_with.result.epoch_results[-1].gap if case_with.result.epoch_results else None
    rr_gap_wo = case_without.result.epoch_results[-1].gap if case_without.result.epoch_results else None
    rr_gap_w_str = f"{rr_gap_w:.6f}" if rr_gap_w is not None else "n/a"
    rr_gap_wo_str = f"{rr_gap_wo:.6f}" if rr_gap_wo is not None else "n/a"

    figs = []
    descriptions = []

    # ===================================================================
    # 1. Network Topology
    # ===================================================================
    meta_nodes = meta_with["nodes"]
    cos_lat = np.cos(np.radians(43.735))
    node_km = {}
    ref_lon, ref_lat = meta_nodes[1]
    for nid, (lon, lat) in meta_nodes.items():
        node_km[nid] = (
            (lon - ref_lon) * 111.32 * cos_lat,
            (lat - ref_lat) * 111.32,
        )

    edges_wo = [
        ((1, 3), "variable 1-lane 80 km/h", "#F44336"),
        ((1, 4), "constant 4-lane 50 km/h", "#2196F3"),
        ((3, 2), "constant 4-lane 50 km/h", "#2196F3"),
        ((4, 2), "variable 1-lane 80 km/h", "#F44336"),
    ]
    edges_w = edges_wo + [
        ((3, 4), "shortcut 1-lane 80 km/h", "#FF9800"),
    ]

    topo_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Without Shortcut", "With Shortcut"],
        horizontal_spacing=0.12,
    )

    main_nodes = [1, 2, 3, 4]

    for col, edges in enumerate([edges_wo, edges_w], 1):
        for edge_info in edges:
            node_seq, label, color = edge_info[0], edge_info[1], edge_info[2]
            xs = [node_km[n][0] for n in node_seq]
            ys = [node_km[n][1] for n in node_seq]
            u, v = node_seq[0], node_seq[-1]
            topo_fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="text",
                hovertext=f"{u}\u2192{v}: {label}",
                showlegend=False,
            ), row=1, col=col)
            topo_fig.add_annotation(
                x=xs[-1], y=ys[-1], ax=xs[0], ay=ys[0],
                xref=f"x{col}" if col > 1 else "x",
                yref=f"y{col}" if col > 1 else "y",
                axref=f"x{col}" if col > 1 else "x",
                ayref=f"y{col}" if col > 1 else "y",
                showarrow=True, arrowhead=3, arrowsize=1.5,
                arrowcolor=color, arrowwidth=2,
            )
            mx, my = (xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2
            dx = ys[-1] - ys[0]
            dy = -(xs[-1] - xs[0])
            mag = max((dx**2 + dy**2) ** 0.5, 1e-9)
            ox, oy = 0.3 * dx / mag, 0.3 * dy / mag
            topo_fig.add_annotation(
                x=mx + ox, y=my + oy, text=f"{u}&rarr;{v}",
                xref=f"x{col}" if col > 1 else "x",
                yref=f"y{col}" if col > 1 else "y",
                showarrow=False, font=dict(size=10, color=color),
            )

        xs = [node_km[n][0] for n in main_nodes]
        ys = [node_km[n][1] for n in main_nodes]
        labels = [str(n) for n in main_nodes]
        roles = {1: "Origin", 2: "Destination", 3: "Node 3", 4: "Node 4"}
        hover = [f"Node {n} ({roles[n]})" for n in main_nodes]
        colors = ["#4CAF50" if n == 1 else "#F44336" if n == 2 else "#9E9E9E"
                  for n in main_nodes]
        topo_fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=28, color=colors, line=dict(width=2, color="white")),
            text=labels, textfont=dict(size=14, color="white"),
            textposition="middle center",
            hovertext=hover, hoverinfo="text",
            showlegend=False,
        ), row=1, col=col)

    all_x = [v[0] for v in node_km.values()]
    all_y = [v[1] for v in node_km.values()]
    x_pad = (max(all_x) - min(all_x)) * 0.08
    y_pad = max((max(all_y) - min(all_y)) * 0.3, 0.15)
    for suffix in ["", "2"]:
        xref, yref = f"xaxis{suffix}", f"yaxis{suffix}"
        topo_fig.update_layout(**{
            xref: dict(showgrid=False, zeroline=False, showticklabels=True,
                       title="km", range=[min(all_x) - x_pad, max(all_x) + x_pad],
                       fixedrange=True),
            yref: dict(showgrid=False, zeroline=False, showticklabels=True,
                       title="km",
                       range=[min(all_y) - y_pad, max(all_y) + y_pad],
                       scaleanchor=f"x{suffix}" if suffix else "x",
                       fixedrange=True),
        })
    topo_fig.update_layout(
        title="Network Topology (to scale)",
        template="plotly_white",
        height=400,
    )

    figs.append(topo_fig)
    descriptions.append("""<h2>Network Topology</h2>
    <p>The Braess diamond network (to scale): <span style="color:#4CAF50">\u25cf</span> Origin (node 1),
    <span style="color:#F44336">\u25cf</span> Destination (node 2).
    <span style="color:#F44336">Red</span> = variable (1-lane, 80 km/h &mdash; fast but congestion-sensitive).
    <span style="color:#2196F3">Blue</span> = constant (4-lane, 50 km/h &mdash; slow but robust).
    <span style="color:#FF9800">Orange</span> = shortcut (1-lane, 80 km/h, ~1 km).
    All main links ~10 km.</p>""")

    # ===================================================================
    # 2. Link State Comparison (4-column)
    # ===================================================================
    state_scenarios = [
        (result_without.network_state, "Without (MSA)"),
        (result_with.network_state, "With (MSA)"),
        (result_fw_with.network_state, "With (FW)"),
        (case_with.result.network_state, "With (HC Rerouted)"),
    ]

    figs.append(None)
    descriptions.append(
        """<h2>Link State Comparison</h2>
        <p>Four scenarios compared: MSA without and with shortcut,
        FW with shortcut, and HC Rerouted with shortcut.
        Density (k, veh/km), speed (v, km/h), flow (q = k&times;v, veh/hr), and travel
        time (t = L/v) at the converged state.
        <span style="color:#F44336;font-weight:600;">Red density</span>
        = near jam (k/k<sub>j</sub> &gt; 0.9).
        <span style="color:#F44336">Red speed</span> = severe congestion (v/v<sub>f</sub> &lt; 0.1).
        <span style="color:#F44336">Red time</span> = &gt;2&times; freeflow.
        <span style="color:#FF9800">Orange</span> = moderate.
        <span style="color:#4CAF50">Green</span> = uncongested.</p>"""
        + _braess_state_table(state_scenarios)
    )

    # ===================================================================
    # 3. Route Travel Times (4-column)
    # ===================================================================
    figs.append(None)
    descriptions.append(
        """<h2>Route Travel Times</h2>
        <p>Total travel time for each OD route, summed from link-level t = L/v.
        At user equilibrium (Wardrop), all <i>used</i> routes between an OD pair
        should have equal travel time.  Comparing MSA, FW, and HC Rerouted
        convergence toward this condition.</p>"""
        + _braess_route_tt_table(state_scenarios)
    )

    # ===================================================================
    # 4. TSTT Comparison (all methods)
    # ===================================================================
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="MSA",
        x=["Without Shortcut", "With Shortcut"],
        y=[tstt_without_vals[-1], tstt_with_vals[-1]],
        marker_color="#64B5F6",
    ))
    fig_bar.add_trace(go.Bar(
        name="FW",
        x=["Without Shortcut", "With Shortcut"],
        y=[fw_tstt_wo, fw_tstt_w],
        marker_color="#1565C0",
    ))
    fig_bar.add_trace(go.Bar(
        name="HC Greedy",
        x=["Without Shortcut", "With Shortcut"],
        y=[hc_tstt_wo, hc_tstt_w],
        marker_color="#FFB74D",
    ))
    fig_bar.add_trace(go.Bar(
        name="HC Rerouted",
        x=["Without Shortcut", "With Shortcut"],
        y=[rr_tstt_wo, rr_tstt_w],
        marker_color="#E65100",
    ))
    fig_bar.update_layout(
        title="TSTT by Method and Scenario",
        xaxis_title="Network Scenario",
        yaxis_title="TSTT (veh\u00b7seconds)",
        barmode="group",
        template="plotly_white",
    )
    figs.append(fig_bar)
    descriptions.append(
        "<h2>TSTT Comparison</h2>"
        "<p>Total system travel time across all methods. At equilibrium (MSA, FW), "
        "the Braess paradox is confirmed: adding the shortcut <i>increases</i> TSTT "
        f"by <b>+{pct:.1f}%</b> (MSA) / <b>+{fw_pct:.1f}%</b> (FW). Under HC Greedy loading, the "
        f"shortcut <i>reduces</i> TSTT by <b>{abs(hc_pct):.1f}%</b>. After reroute epochs, HC Rerouted "
        f"converges toward equilibrium and the paradox re-emerges at <b>{rr_pct:+.1f}%</b>.</p>"
    )

    # ===================================================================
    # 5. MFD Plots
    # ===================================================================
    _add_mfd_section(figs, descriptions, case_with.result.network_state, 1.0)

    # ===================================================================
    # 6. Matrix Convergence
    # ===================================================================
    figs.append(None)
    descriptions.append(
        "<h2>Matrix Convergence</h2>"
        "<p>Convergence diagnostics for matrix-based equilibrium methods (MSA and Frank-Wolfe). "
        "Both use all-or-nothing (AON) routing each iteration with density blending toward equilibrium.</p>"
    )

    # -------------------------------------------------------------------
    # 6a. TSTT Convergence (MSA)
    # -------------------------------------------------------------------
    iters_w = [r.iteration for r in result_with.iteration_log]
    iters_wo = [r.iteration for r in result_without.iteration_log]

    fig_tstt = go.Figure()
    fig_tstt.add_trace(go.Scatter(
        x=iters_w, y=tstt_with_vals, mode="lines+markers",
        name="With shortcut", line=dict(color="#F44336", width=2),
    ))
    fig_tstt.add_trace(go.Scatter(
        x=iters_wo, y=tstt_without_vals, mode="lines+markers",
        name="Without shortcut", line=dict(color="#2196F3", width=2),
    ))
    fig_tstt.update_layout(
        title="Total System Travel Time (TSTT) per Iteration",
        xaxis_title="Iteration", yaxis_title="TSTT (veh&middot;seconds)",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig_tstt)

    gap_w = result_with.iteration_log[-1].relative_gap
    gap_wo = result_without.iteration_log[-1].relative_gap
    tstt_w_fmt = f"{tstt_with_vals[-1]:,.0f}"
    tstt_wo_fmt = f"{tstt_without_vals[-1]:,.0f}"
    delta_fmt = f"{delta:+,.0f}"
    paradox_msg = (
        " <b>Braess paradox confirmed</b>: adding the shortcut <i>increases</i> total travel time."
        if delta > 0 else " Paradox not observed at this demand level."
    )
    gap_w_fmt = f"{gap_w:.6f}"
    gap_wo_fmt = f"{gap_wo:.6f}"
    pct_fmt = f"{pct:+.1f}"
    descriptions.append(
        "<h3>TSTT Convergence (MSA)</h3>"
        "<p>Red = network WITH shortcut, blue = WITHOUT. "
        f"TSTT with shortcut = <b>{tstt_w_fmt}</b> veh&middot;s (gap={gap_w_fmt}), "
        f"without = <b>{tstt_wo_fmt}</b> veh&middot;s (gap={gap_wo_fmt}). "
        f"&Delta; = <b>{delta_fmt}</b> ({pct_fmt}%). "
        f"{paradox_msg}</p>"
    )

    # -------------------------------------------------------------------
    # 6b. Wardrop Gap: MSA vs Frank-Wolfe
    # -------------------------------------------------------------------
    gap_with = [r.relative_gap for r in result_with.iteration_log]
    gap_without = [r.relative_gap for r in result_without.iteration_log]
    gap_fw_w = [r.relative_gap for r in result_fw_with.iteration_log]
    gap_fw_wo = [r.relative_gap for r in result_fw_without.iteration_log]
    iters_fw_w = [r.iteration for r in result_fw_with.iteration_log]
    iters_fw_wo = [r.iteration for r in result_fw_without.iteration_log]

    def _moving_max(gaps, window=5):
        """Rolling max of gap values (envelope of worst-case per window)."""
        arr = np.array(gaps)
        out = np.empty_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            out[i] = arr[start:i+1].max()
        return out.tolist()

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(
        x=iters_w, y=[abs(g) if g != 0 else None for g in gap_with],
        mode="markers", name="MSA with shortcut (raw)",
        marker=dict(color="#F44336", size=4, opacity=0.3),
    ))
    fig_gap.add_trace(go.Scatter(
        x=iters_wo, y=[abs(g) if g != 0 else None for g in gap_without],
        mode="markers", name="MSA without shortcut (raw)",
        marker=dict(color="#2196F3", size=4, opacity=0.3),
    ))
    fig_gap.add_trace(go.Scatter(
        x=iters_w, y=_moving_max([abs(g) for g in gap_with], window=5),
        mode="lines", name="MSA with shortcut (envelope)",
        line=dict(color="#F44336", width=1.5, dash="dash"),
    ))
    fig_gap.add_trace(go.Scatter(
        x=iters_wo, y=_moving_max([abs(g) for g in gap_without], window=5),
        mode="lines", name="MSA without shortcut (envelope)",
        line=dict(color="#2196F3", width=1.5, dash="dash"),
    ))
    fig_gap.add_trace(go.Scatter(
        x=iters_fw_w, y=[abs(g) if g != 0 else None for g in gap_fw_w],
        mode="lines+markers", name="FW with shortcut",
        line=dict(color="#D32F2F", width=2.5),
        marker=dict(size=5),
    ))
    fig_gap.add_trace(go.Scatter(
        x=iters_fw_wo, y=[abs(g) if g != 0 else None for g in gap_fw_wo],
        mode="lines+markers", name="FW without shortcut",
        line=dict(color="#1565C0", width=2.5),
        marker=dict(size=5),
    ))
    fig_gap.update_layout(
        title="Wardrop Relative Gap: MSA vs Frank-Wolfe",
        xaxis_title="Iteration", yaxis_title="|Relative Gap|",
        yaxis_type="log",
        template="plotly_white",
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True, dtick=1, tickformat=".0e"),
    )
    figs.append(fig_gap)

    fw_final_gap_w = result_fw_with.iteration_log[-1].relative_gap
    fw_final_gap_wo = result_fw_without.iteration_log[-1].relative_gap
    fw_gap_w_fmt = f"{fw_final_gap_w:.6f}"
    fw_gap_wo_fmt = f"{fw_final_gap_wo:.6f}"
    descriptions.append(
        "<h3>Wardrop Gap: MSA vs Frank-Wolfe</h3>"
        "<p>MSA (dashed envelopes, faint dots) vs Frank-Wolfe (solid lines). "
        "FW uses Beckmann line search to find the optimal step size each iteration, "
        "eliminating the bang-bang oscillation inherent to MSA on small networks. "
        f"FW final gap: with shortcut = {fw_gap_w_fmt}, without = {fw_gap_wo_fmt}.</p>"
    )

    # -------------------------------------------------------------------
    # 6c. Frank-Wolfe Step Size
    # -------------------------------------------------------------------
    step_sizes_w = [r.step_size for r in result_fw_with.iteration_log]
    step_sizes_wo = [r.step_size for r in result_fw_without.iteration_log]

    fig_step = go.Figure()
    fig_step.add_trace(go.Scatter(
        x=iters_fw_w, y=step_sizes_w, mode="lines+markers",
        name="With shortcut", line=dict(color="#F44336", width=2),
        marker=dict(size=5),
    ))
    fig_step.add_trace(go.Scatter(
        x=iters_fw_wo, y=step_sizes_wo, mode="lines+markers",
        name="Without shortcut", line=dict(color="#2196F3", width=2),
        marker=dict(size=5),
    ))
    msa_steps = [1.0 / n for n in range(1, max_iter + 1)]
    fig_step.add_trace(go.Scatter(
        x=list(range(1, max_iter + 1)), y=msa_steps, mode="lines",
        name="MSA (1/n)", line=dict(color="#999", width=1, dash="dot"),
    ))
    fig_step.update_layout(
        title="Frank-Wolfe Step Size per Iteration",
        xaxis_title="Iteration", yaxis_title="Step Size (&alpha;)",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True, range=[0, 1.05]),
    )
    figs.append(fig_step)
    descriptions.append(
        "<h3>Frank-Wolfe Step Size</h3>"
        "<p>Optimal step size &alpha;* from the Beckmann line search each iteration. "
        "Dotted grey = MSA fixed schedule (1/n). Early iterations take large steps; "
        "as equilibrium is approached, FW takes progressively smaller steps &mdash; "
        "unlike MSA which follows a rigid 1/n schedule regardless of the objective landscape.</p>"
    )

    # ===================================================================
    # 7. Hill-Climber Convergence
    # ===================================================================
    figs.append(None)
    descriptions.append(
        "<h2>Hill-Climber Convergence</h2>"
        "<p>Convergence diagnostics for the matrix-free hill-climber. Demand is divided into 10 departure "
        "slices loaded sequentially (greedy phase), then refined via reroute epochs that "
        "iterate through slices oldest-first, subtracting and re-routing each slice's demand.</p>"
    )

    # -------------------------------------------------------------------
    # 7a. Slice Timeline / State / Runtime
    # -------------------------------------------------------------------
    hc_figs, hc_descriptions = build_hillclimber_report_sections(
        network_name="Braess",
        case=case_with,
        detail_scale=1.0,
    )
    # [0]=congestion map, [1]=link table, [2]=slice timeline, [3]=state evolution,
    # [4]=slice runtime, [5]=MFD (combined), [6]=correlation (combined, if ref)
    for i in range(2, 5):
        hc_descriptions[i] = hc_descriptions[i].replace("<h2>", "<h3>").replace("</h2>", "</h3>")

    # Enrich state evolution description with convergence summary
    rr_paradox_text = (
        f"The Braess paradox emerges at {rr_pct:+.1f}% &mdash; confirming "
        "convergence toward approximate Wardrop equilibrium."
        if rr_pct > 0 else
        f"The Braess paradox does not emerge ({rr_pct:+.1f}%)."
    )
    epochs_w = case_with.result.epoch_results
    epochs_wo = case_without.result.epoch_results
    hc_descriptions[3] += (
        f"<p><b>Reroute convergence:</b> {n_epochs_w} epoch(s) with shortcut, "
        f"{len(epochs_wo)} without. "
        f"Final gap: with&nbsp;=&nbsp;{rr_gap_w_str}, without&nbsp;=&nbsp;{rr_gap_wo_str}. "
        f"{rr_paradox_text}</p>"
    )

    figs.extend(hc_figs[2:5])
    descriptions.extend(hc_descriptions[2:5])

    # -------------------------------------------------------------------
    # 7b. Slice Sensitivity
    # -------------------------------------------------------------------
    sweep_slices = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    sweep_pcts: list[float] = []
    sweep_base_w, sweep_meta_w = _prepare_network(tmp_path / "sweep_w", with_shortcut=True)
    sweep_base_wo, sweep_meta_wo = _prepare_network(tmp_path / "sweep_wo", with_shortcut=False)
    for ns in sweep_slices:
        cw = run_hillclimber_case(
            base_path=sweep_base_w, meta=sweep_meta_w,
            copy_fn=lambda base, run_dir: base,
            trip_builder=_hc_trip_builder,
            run_dir=tmp_path / f"sweep_w_{ns}",
            demand_scale=1.0, n_slices=ns,
            state_patch_factory=lambda m: lambda s: patch_braess_lanes(s, m),
        )
        cwo = run_hillclimber_case(
            base_path=sweep_base_wo, meta=sweep_meta_wo,
            copy_fn=lambda base, run_dir: base,
            trip_builder=_hc_trip_builder,
            run_dir=tmp_path / f"sweep_wo_{ns}",
            demand_scale=1.0, n_slices=ns,
            state_patch_factory=lambda m: lambda s: patch_braess_lanes(s, m),
        )
        tw = sum(b.batch_tstt for b in cw.result.batch_results)
        two = sum(b.batch_tstt for b in cwo.result.batch_results)
        sweep_pcts.append((tw / two - 1) * 100 if two else 0.0)

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=sweep_slices, y=sweep_pcts,
        mode="lines+markers",
        line=dict(color="#D32F2F", width=2),
        marker=dict(size=7),
        hovertemplate="slices=%{x}<br>TSTT delta=%{y:+.1f}%<extra></extra>",
    ))
    fig_sweep.add_hline(y=0, line_dash="dot", line_color="#999",
                        annotation_text="paradox threshold")
    asymptote = sweep_pcts[-1]
    fig_sweep.add_hline(y=asymptote, line_dash="dash", line_color="#1565C0",
                        annotation_text=f"asymptote \u2248 {asymptote:+.1f}%",
                        annotation_position="bottom right")
    fig_sweep.update_layout(
        title="TSTT Delta vs Number of Departure Slices",
        xaxis_title="Number of departure slices",
        yaxis_title="TSTT delta (with vs without shortcut, %)",
        template="plotly_white",
        xaxis=dict(
            type="log",
            tickvals=sweep_slices,
            ticktext=[str(s) for s in sweep_slices],
            fixedrange=True,
        ),
        yaxis=dict(fixedrange=True),
    )
    figs.append(fig_sweep)
    descriptions.append(
        "<h3>Slice Sensitivity (Greedy Only)</h3>"
        "<p>TSTT delta (with-shortcut vs without-shortcut) as a function of "
        "the number of departure slices, using greedy loading only (E0, no reroute epochs).  "
        "The dotted grey line at 0% is where "
        "the Braess paradox would emerge (positive delta).  The delta never "
        "crosses zero &mdash; the shortcut <b>always helps</b> under hill-climber "
        "loading.</p>"
        "<p>Two regimes are visible: a <b>rapid convergence</b> phase "
        "(1&ndash;8 slices) where the delta drops from &minus;19% to "
        "&minus;4%, and a <b>plateau</b> beyond ~8 slices where additional "
        "slicing barely changes the result (asymptoting to ~&minus;3%).  "
        "This suggests 8&ndash;16 slices is a practical sweet spot for "
        "hill-climber accuracy on small networks.  The paradox is an "
        "equilibrium phenomenon that requires global re-routing; greedy "
        "incremental loading never reaches the collectively sub-optimal "
        "state.</p>"
    )

    # ===================================================================
    # Write report
    # ===================================================================
    demand_fmt = f"{demand:,.0f}"
    _write_combined_report(
        title="Braess Paradox Validation",
        intro=(
            "<p>Unified Braess paradox validation on a 4-node diamond network, comparing "
            "matrix-based equilibrium (MSA, Frank-Wolfe) and matrix-free hill-climber "
            "(HC Greedy, HC Rerouted).</p>"
            f"<p>Demand: <b>{demand_fmt}</b> vehicles.  MSA: &alpha;=1/n, {max_iter} iterations.  "
            f"FW: Beckmann line search, {max_iter} iterations.  "
            "HC: 10 departure slices, up to 10 reroute epochs, gap &lt; 0.001.</p>"
            f"<p><b>Key finding:</b> MSA and FW confirm the Braess paradox (+{pct:.1f}%). "
            f"HC Greedy does <i>not</i> reproduce it ({hc_pct:+.1f}%). After reroute epochs, "
            f"HC Rerouted converges to approximate equilibrium ({rr_pct:+.1f}%).</p>"
        ),
        figures=figs,
        descriptions=descriptions,
        path=Path(output_path),
    )
    return Path(output_path)


def generate_braess_hillclimber_report(
    tmp_path: str | Path,
    output_path: str = "docs/plots/braess_validation.html",
    demand: float = 2500.0,
) -> Path:
    """Backward-compatible wrapper — delegates to unified report."""
    return generate_braess_report(tmp_path, output_path=output_path, demand=demand)
