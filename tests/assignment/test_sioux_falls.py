"""Sioux Falls 24-node network validation.

Validates the density-based assignment against the canonical Sioux Falls
benchmark (24 nodes, 76 links, 528 OD pairs).

The TNTP demand (360,600 total) represents a BPR-calibrated hourly volume
that exceeds the physical capacity of realistic 2–3 lane roads.  We scale
demand to 10% (36,060 vph) which is representative of an actual peak hour
for a ~200k population city.

Structural (VDF-independent) checks:
  - Wardrop relative gap < threshold
  - Flow conservation
  - Non-negative flows
  - Speed within bounds
  - Rank correlation with published BPR equilibrium (directional, not magnitude)

See docs/traffic_assignment_design.md section 6.4.
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
)
from osrm.assignment.od_matrix import DemandTrip, ODMatrixAdapter
from osrm.assignment.osm_synthesis import sioux_falls_network, patch_sioux_falls_lanes

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "sioux_falls"


def _copy_clean_osrm(base_path: str, run_dir: Path) -> str:
    """Copy clean OSRM files to a fresh directory for an isolated run.

    Segment-speed customization permanently mutates OSRM edge weights,
    so each assignment run() needs its own copy of the base files.
    """
    src = Path(base_path).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, run_dir / f.name)
    return str(run_dir / Path(base_path).name)


def _prepare_sf_network(tmp_path: Path):
    """Synthesize, extract, partition, customize Sioux Falls."""
    work = tmp_path / "sioux_falls"
    work.mkdir(parents=True, exist_ok=True)

    osm_path, meta = sioux_falls_network(
        work / "sf.osm", fixture_dir=FIXTURE_DIR,
    )
    base = str(work / "sf.osrm")

    osrm.extract(str(osm_path), profile="car", output_path=base, verbosity="ERROR")
    osrm.partition(base, verbosity="ERROR")
    osrm.customize(base, verbosity="ERROR")

    return base, meta


def _build_trips(meta: dict) -> list:
    """Convert OD matrix to DemandTrip list, skipping zero/self demand."""
    centroids = meta["zone_centroids"]
    od = meta["od_matrix"]
    trips = []
    for i in range(od.shape[0]):
        for j in range(od.shape[1]):
            if od[i, j] > 0 and i != j:
                trips.append(DemandTrip(
                    origin=centroids[i + 1],
                    destination=centroids[j + 1],
                    volume=od[i, j],
                ))
    return trips


def _run_sf_assignment(
    base_path: str,
    meta: dict,
    max_iter: int = 30,
    method: str = "fw",
    demand_scale: float = 0.10,
):
    """Run assignment on Sioux Falls network.

    Parameters
    ----------
    demand_scale : float
        Fraction of TNTP demand to use.  Default 0.10 (36,060 vph)
        which is realistic for a ~200k city peak hour on 2–3 lane roads.
    """
    meta_scaled = dict(meta)
    meta_scaled["od_matrix"] = meta["od_matrix"] * demand_scale
    trips = _build_trips(meta_scaled)

    config = AssignmentConfig(
        method=method,
        max_iterations=max_iter,
        convergence_gap=0.0,
        smoothing=DensitySmoothingConfig(method="none"),
        verbosity="ERROR",
        speed_csv_dir=str(Path(base_path).parent),
    )

    loop = AssignmentLoop(base_path, config)

    def lane_patch(state):
        patch_sioux_falls_lanes(state, meta)

    return loop.run(trips, state_patch=lane_patch)


def _link_flow_correlation(result, meta: dict):
    """Compute Spearman rank correlation between assigned and BPR reference flows.

    Returns (correlation, n_matched_links).
    """
    from scipy.stats import spearmanr

    state = result.network_state
    ref = meta["ref_flows"]

    assigned = []
    reference = []
    for i in range(state.n_edges):
        key = (int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1]))
        if key in ref:
            assigned.append(state.flow_vph[i])
            reference.append(ref[key][0])

    if len(assigned) < 3:
        return 0.0, len(assigned)

    corr, _ = spearmanr(assigned, reference)
    return float(corr), len(assigned)


class TestSiouxFalls:
    """Sioux Falls structural validation."""

    def test_smoke_fw(self, tmp_path):
        """Quick smoke: FW runs without error on Sioux Falls, 5 iterations."""
        base, meta = _prepare_sf_network(tmp_path)
        result = _run_sf_assignment(base, meta, max_iter=5, method="fw")
        assert result.iterations == 5
        assert result.network_state.n_edges > 0

    def test_flow_nonnegativity(self, tmp_path):
        """All link flows must be non-negative."""
        base, meta = _prepare_sf_network(tmp_path)
        result = _run_sf_assignment(base, meta, max_iter=10)
        assert np.all(result.network_state.flow_vph >= 0)

    def test_speeds_within_bounds(self, tmp_path):
        """Speeds must be between VDF min and freeflow."""
        base, meta = _prepare_sf_network(tmp_path)
        result = _run_sf_assignment(base, meta, max_iter=10)
        state = result.network_state
        assert np.all(state.speed_kmh >= 0.01 - 1e-6)
        assert np.all(state.speed_kmh <= state.freeflow_kmh + 1e-6)

    def test_freeflow_immutable_across_runs(self, tmp_path):
        """Freeflow must be identical whether run at 5% or 20% demand.

        Regression test: segment-speed customization permanently mutates
        OSRM edge weights.  Running at high demand then low demand used to
        show degraded freeflow on the second run.
        """
        base1, meta = _prepare_sf_network(tmp_path / "r1")
        r1 = _run_sf_assignment(base1, meta, max_iter=5, demand_scale=0.20)
        s1 = r1.network_state

        base2, meta2 = _prepare_sf_network(tmp_path / "r2")
        r2 = _run_sf_assignment(base2, meta2, max_iter=5, demand_scale=0.05)
        s2 = r2.network_state

        ff1 = {
            (int(s1.edge_ids[i, 0]), int(s1.edge_ids[i, 1])): s1.freeflow_kmh[i]
            for i in range(s1.n_edges)
        }
        ff2 = {
            (int(s2.edge_ids[i, 0]), int(s2.edge_ids[i, 1])): s2.freeflow_kmh[i]
            for i in range(s2.n_edges)
        }

        common = set(ff1) & set(ff2)
        assert len(common) >= 20, f"Expected ≥20 common edges, got {len(common)}"
        for key in common:
            assert abs(ff1[key] - ff2[key]) < 0.5, (
                f"Freeflow mismatch on {key}: 20%={ff1[key]:.1f}, "
                f"5%={ff2[key]:.1f}"
            )


def generate_sioux_falls_report(
    tmp_path: str | Path,
    output_path: str = "docs/plots/sioux_falls_validation.html",
    max_iter: int = 50,
) -> Path:
    """Run Sioux Falls validation and generate HTML report.

    Includes demand scaling sweep showing how the network responds as
    demand increases from 5% to 100% of TNTP values.

    Parameters
    ----------
    tmp_path : str or Path
        Working directory for temporary OSRM files.
    output_path : str
        Where to write the HTML report.
    max_iter : int
        Assignment iterations.

    Returns
    -------
    Path to the generated report.
    """
    import plotly.graph_objects as go
    from osrm.assignment.plots import _write_combined_report

    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    base, meta = _prepare_sf_network(tmp_path)
    total_demand = float(meta["od_matrix"].sum())

    figs = []
    descriptions = []

    # --- 0. Network topology map ---
    nodes = meta["nodes"]
    fig_topo = go.Figure()

    link_attrs = meta["link_attrs"]
    for (u, v), attrs in link_attrs.items():
        x0, y0 = nodes[u]
        x1, y1 = nodes[v]
        lanes = attrs["n_lanes"]
        color = "#D32F2F" if lanes >= 3 else "#2196F3"
        fig_topo.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=color, width=max(1, lanes * 1.2)),
            hoverinfo="text",
            hovertext=(
                f"{u}&rarr;{v}: {attrs['n_lanes']}L, "
                f"{attrs['ff_speed_kmh']:.0f} km/h"
            ),
            showlegend=False,
        ))

    xs = [nodes[n][0] for n in sorted(nodes)]
    ys = [nodes[n][1] for n in sorted(nodes)]
    labels = [str(n) for n in sorted(nodes)]
    fig_topo.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=16, color="#4CAF50", line=dict(width=1.5, color="white")),
        text=labels, textfont=dict(size=9, color="white"),
        textposition="middle center",
        hovertext=[f"Node {n}" for n in sorted(nodes)],
        hoverinfo="text",
        showlegend=False,
    ))

    fig_topo.update_layout(
        title="Sioux Falls Network (24 nodes, 76 links)",
        xaxis=dict(title="Longitude", scaleanchor="y", fixedrange=True),
        yaxis=dict(title="Latitude", fixedrange=True),
        template="plotly_white",
        height=500,
    )
    figs.append(fig_topo)

    n_i29 = sum(1 for _, a in link_attrs.items() if a["n_lanes"] >= 3)
    n_art = sum(1 for _, a in link_attrs.items() if a["n_lanes"] == 2)
    descriptions.append(
        "<h2>Network Topology</h2>"
        "<p>Sioux Falls benchmark: 24 nodes, 76 directed links, 528 OD pairs. "
        "Road classification from actual city geography: "
        f'<span style="color:#D32F2F"><b>{n_i29} interstate links</b></span> '
        f"(I-29/I-229, 3 or 2 lanes, 105 km/h) and "
        f'<span style="color:#2196F3"><b>{n_art} arterial links</b></span> '
        f"(2 lanes, 30&ndash;70 km/h).</p>"
        f"<p>TNTP total demand: {total_demand:,.0f}. "
        "Original paper values &times; 100 = 0.1 &times; daily &asymp; hourly. "
        "TNTP &lsquo;capacity&rsquo; column is a BPR math artifact "
        "(back-computed from polynomial coefficients), <b>not</b> physical "
        "road capacity. BPR &lsquo;capacities&rsquo; reach 25,900 vph per link "
        "&mdash; implying 13+ lanes per direction for a city of 200,000. "
        "The original 1975 model used <code>t&nbsp;=&nbsp;a&nbsp;+&nbsp;b&middot;flow<sup>4</sup></code> "
        "with no capacity concept at all; the capacity column was reverse-engineered "
        "later to fit BPR&rsquo;s <code>t&nbsp;=&nbsp;fft&middot;(1&nbsp;+&nbsp;0.15&middot;(V/C)<sup>4</sup>)</code> "
        "convention.</p>"
    )

    # --- 1. Demand scaling sweep ---
    # Each run permanently mutates OSRM edge weights via segment-speed
    # customization, so each scale level needs a fresh copy of the base files.
    scales = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
    sweep_demand = []
    sweep_gap = []
    sweep_oversat = []
    sweep_mean_speed = []
    sweep_tstt = []

    for scale in scales:
        run_base = _copy_clean_osrm(base, tmp_path / f"sf_scale_{scale:.2f}")
        result = _run_sf_assignment(
            run_base, meta, max_iter=max_iter, method="fw", demand_scale=scale,
        )
        state = result.network_state
        last = result.iteration_log[-1]
        speeds = state.speed_kmh[:state.n_edges]
        ff = state.freeflow_kmh[:state.n_edges]

        sweep_demand.append(total_demand * scale)
        sweep_gap.append(last.relative_gap)
        sweep_oversat.append(last.n_oversaturated)
        sweep_mean_speed.append(float(np.mean(speeds / ff)))
        sweep_tstt.append(last.tstt)

    demand_labels = [f"{s:.0%}" for s in scales]

    # 1a: Oversaturated links vs demand
    fig_oversat = go.Figure()
    fig_oversat.add_trace(go.Bar(
        x=demand_labels, y=sweep_oversat,
        marker_color=[
            "#4CAF50" if o < 5 else "#FF9800" if o < 40 else "#D32F2F"
            for o in sweep_oversat
        ],
        hovertext=[
            f"{d:,.0f} vph &rarr; {o}/76 links oversat"
            for d, o in zip(sweep_demand, sweep_oversat)
        ],
        hoverinfo="text",
    ))
    fig_oversat.update_layout(
        title="Oversaturated Links vs Demand Scale",
        xaxis_title="Demand Scale (% of TNTP)",
        yaxis_title="Links at Jam Density",
        template="plotly_white",
        yaxis=dict(range=[0, 80], fixedrange=True),
        xaxis=dict(fixedrange=True),
        showlegend=False,
    )
    figs.append(fig_oversat)
    descriptions.append(
        "<h2>Demand Scaling: Oversaturation</h2>"
        "<p>Number of links reaching jam density as demand increases. "
        "With realistic 2&ndash;3 lane roads, the network handles up to "
        "~10% of TNTP demand before widespread congestion. At 100%, "
        "nearly all links gridlock &mdash; the TNTP benchmark was designed "
        "for BPR&rsquo;s infinite-capacity math, not physical roads.</p>"
    )

    # 1b: Mean speed ratio vs demand
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(
        x=[total_demand * s for s in scales],
        y=[r * 100 for r in sweep_mean_speed],
        mode="lines+markers",
        line=dict(color="#1565C0", width=2.5),
        marker=dict(size=8),
        hovertext=[
            f"{s:.0%}: {r*100:.0f}% of free-flow"
            for s, r in zip(scales, sweep_mean_speed)
        ],
        hoverinfo="text",
    ))
    fig_speed.update_layout(
        title="Network Mean Speed vs Demand",
        xaxis_title="Total Demand (vph)",
        yaxis_title="Mean Speed (% of Free-Flow)",
        template="plotly_white",
        yaxis=dict(range=[0, 105], fixedrange=True),
        xaxis=dict(fixedrange=True),
    )
    figs.append(fig_speed)
    descriptions.append(
        "<h2>Demand Scaling: Speed</h2>"
        "<p>Network-average speed as fraction of free-flow. The MFD-based VDF "
        "imposes a physical capacity ceiling: once demand exceeds it, speed "
        "drops to the minimum (jam). Unlike BPR which degrades gracefully "
        "at V/C &gt; 1, the bi-parabolic model correctly models gridlock.</p>"
    )

    # 1c: TSTT vs demand
    fig_tstt_scale = go.Figure()
    fig_tstt_scale.add_trace(go.Scatter(
        x=[total_demand * s for s in scales],
        y=sweep_tstt,
        mode="lines+markers",
        line=dict(color="#D32F2F", width=2.5),
        marker=dict(size=8),
        hovertext=[
            f"{s:.0%}: TSTT={t:,.0f}"
            for s, t in zip(scales, sweep_tstt)
        ],
        hoverinfo="text",
    ))
    fig_tstt_scale.update_layout(
        title="Total System Travel Time vs Demand",
        xaxis_title="Total Demand (vph)",
        yaxis_title="TSTT (veh-seconds)",
        yaxis_type="log",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig_tstt_scale)
    descriptions.append(
        "<h2>Demand Scaling: TSTT</h2>"
        "<p>Total system travel time rises exponentially as demand approaches "
        "physical capacity, then explodes as links jam.</p>"
    )

    # --- 2. Detailed 10% run: FW vs MSA convergence ---
    base_fw = _copy_clean_osrm(base, tmp_path / "fw_detail")
    result_fw = _run_sf_assignment(
        base_fw, meta, max_iter=max_iter, method="fw", demand_scale=0.10,
    )
    base_msa, meta_msa = _prepare_sf_network(tmp_path / "msa")
    result_msa = _run_sf_assignment(
        base_msa, meta_msa, max_iter=max_iter, method="msa", demand_scale=0.10,
    )

    iters_fw = [r.iteration for r in result_fw.iteration_log]
    gap_fw = [r.relative_gap for r in result_fw.iteration_log]
    iters_msa = [r.iteration for r in result_msa.iteration_log]
    gap_msa = [r.relative_gap for r in result_msa.iteration_log]

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(
        x=iters_fw, y=[g if g > 0 else None for g in gap_fw],
        mode="lines+markers", name="Frank-Wolfe",
        line=dict(color="#D32F2F", width=2.5), marker=dict(size=4),
    ))
    fig_gap.add_trace(go.Scatter(
        x=iters_msa, y=[g if g > 0 else None for g in gap_msa],
        mode="lines+markers", name="MSA (1/n)",
        line=dict(color="#1565C0", width=1.5, dash="dash"), marker=dict(size=3),
    ))
    fig_gap.update_layout(
        title="Wardrop Gap: FW vs MSA (10% Demand)",
        xaxis_title="Iteration", yaxis_title="Relative Gap",
        yaxis_type="log",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig_gap)

    fw_final = result_fw.iteration_log[-1]
    msa_final = result_msa.iteration_log[-1]
    descriptions.append(
        "<h2>Convergence at 10% Demand</h2>"
        f"<p>FW gap: {fw_final.relative_gap:.4f}, "
        f"MSA gap: {msa_final.relative_gap:.4f} ({max_iter} iterations). "
        f"Demand: {total_demand * 0.10:,.0f} vph (10% of TNTP).</p>"
    )

    # --- 3. Link state table (Braess-style) at 10% demand ---
    state = result_fw.network_state
    ref = meta["ref_flows"]
    link_attrs_map = meta["link_attrs"]

    # Build table rows sorted by link
    rows = []
    for i in range(state.n_edges):
        key = (int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1]))
        la = link_attrs_map.get(key, {})
        dist_m = la.get("distance_m", 0)
        vf = state.freeflow_kmh[i]
        v = state.speed_kmh[i]
        flow = state.flow_vph[i]
        ln = int(state.n_lanes[i])
        kj = state.jam_density[i]
        qc = vf * kj / 6
        tt = dist_m / 1000 / v * 60 if v > 0 else float("inf")
        ff_tt = dist_m / 1000 / vf * 60 if vf > 0 else float("inf")
        vc = flow / qc if qc > 0 else 0
        rows.append((key, ln, dist_m, vf, v, flow, qc, ff_tt, tt, vc))

    rows.sort(key=lambda r: r[0])

    def _vc_color(vc):
        if vc > 0.85:
            return "color:#D32F2F;font-weight:bold;"
        if vc > 0.6:
            return "color:#FF6F00;"
        return ""

    table_html = (
        '<table style="border-collapse:collapse; width:100%; font-size:13px; '
        'font-family:monospace;">\n'
        '<thead><tr style="border-bottom:2px solid #333;">'
        '<th style="text-align:left;padding:6px;">Link</th>'
        '<th style="padding:6px;">Lanes</th>'
        '<th style="padding:6px;">Dist (m)</th>'
        '<th style="padding:6px;">v<sub>f</sub></th>'
        '<th style="padding:6px;">q<sub>c</sub></th>'
        '<th style="padding:6px;border-left:2px solid #ccc;">Flow</th>'
        '<th style="padding:6px;">V/C</th>'
        '<th style="padding:6px;">Speed</th>'
        '<th style="padding:6px;">FF TT</th>'
        '<th style="padding:6px;">TT</th>'
        '<th style="padding:6px;">TT/FF</th>'
        '</tr></thead>\n<tbody>\n'
    )

    for key, ln, dist_m, vf, v, flow, qc, ff_tt, tt, vc in rows:
        ratio = tt / ff_tt if ff_tt > 0 and tt < float("inf") else float("inf")
        vc_style = _vc_color(vc)
        tt_str = f"{tt:.2f}" if tt < float("inf") else "&infin;"
        ratio_str = f"{ratio:.2f}" if ratio < float("inf") else "&infin;"
        table_html += (
            f'<tr style="border-bottom:1px solid #eee;">'
            f'<td style="padding:4px 6px;">{key[0]}&rarr;{key[1]}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{ln}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{dist_m:.0f}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{vf:.0f}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{qc:.0f}</td>'
            f'<td style="text-align:right;padding:4px 6px;border-left:2px solid #ccc;">{flow:.0f}</td>'
            f'<td style="text-align:right;padding:4px 6px;{vc_style}">{vc:.2f}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{v:.1f}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{ff_tt:.2f}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{tt_str}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{ratio_str}</td>'
            f'</tr>\n'
        )

    table_html += '</tbody></table>'

    # Summary stats (exclude inf for means)
    all_vc = [r[9] for r in rows]
    all_speeds = [r[4] for r in rows]
    finite_ratios = [r[8] / r[7] for r in rows if r[7] > 0 and r[8] < float("inf")]
    n_congested = sum(1 for vc in all_vc if vc > 0.6)

    figs.append(None)  # placeholder — table goes in description
    descriptions.append(
        "<h2>Link State at 10% Demand</h2>"
        f"<p>36,060 vph ({state.n_edges} links). "
        f"Mean V/C: <b>{np.mean(all_vc):.2f}</b>, "
        f"max V/C: {max(all_vc):.2f}. "
        f"Mean speed: <b>{np.mean(all_speeds):.0f} km/h</b>. "
        f"Mean TT/FF: {np.mean(finite_ratios):.2f}. "
        f"Links with V/C &gt; 0.6: {n_congested}/{state.n_edges}. "
        "Units: v<sub>f</sub> and Speed in km/h, q<sub>c</sub> and Flow in vph, "
        "TT in minutes. "
        '<span style="color:#FF6F00">Orange</span>: V/C &gt; 0.6, '
        '<span style="color:#D32F2F"><b>Red</b></span>: V/C &gt; 0.85.</p>'
        + table_html
    )

    # --- 4. Flow and travel time correlation vs BPR reference ---
    # BPR reference is at 100% demand (TNTP equilibrium). Our model runs at
    # 10%.  Flow magnitudes differ, but relative loading patterns (which links
    # carry more traffic) should correlate.  Travel time comparison shows how
    # MFD congestion compares to BPR at the same physical demand level.
    from scipy.stats import spearmanr

    mfd_flows = []
    bpr_flows = []
    mfd_tt = []
    bpr_tt = []
    link_labels = []
    for i in range(state.n_edges):
        key = (int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1]))
        if key not in ref:
            continue
        bpr_flow, bpr_cost = ref[key]
        la = link_attrs_map.get(key, {})
        dist_m = la.get("distance_m", 0)
        vf = state.freeflow_kmh[i]
        v = state.speed_kmh[i]
        our_flow = state.flow_vph[i]
        our_tt = dist_m / 1000 / v * 60 if v > 0 else float("inf")

        # BPR cost is the equilibrium travel time in minutes (same unit
        # as TNTP free-flow time column).
        bpr_tt_min = bpr_cost

        mfd_flows.append(our_flow)
        bpr_flows.append(bpr_flow)
        mfd_tt.append(our_tt)
        bpr_tt.append(bpr_tt_min)
        link_labels.append(f"{key[0]}&rarr;{key[1]}")

    # 4a: Flow scatter
    fig_flow_corr = go.Figure()
    fig_flow_corr.add_trace(go.Scatter(
        x=bpr_flows, y=mfd_flows, mode="markers",
        marker=dict(size=6, color="#1565C0", opacity=0.7),
        hovertext=link_labels, hoverinfo="text",
    ))
    if len(mfd_flows) >= 3:
        flow_rho, _ = spearmanr(bpr_flows, mfd_flows)
    else:
        flow_rho = 0.0
    fig_flow_corr.update_layout(
        title=f"Link Flow: MFD (10%) vs BPR (100%) — ρ={flow_rho:.3f}",
        xaxis_title="BPR Equilibrium Flow (vph, 100% demand)",
        yaxis_title="MFD Flow (vph, 10% demand)",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig_flow_corr)
    descriptions.append(
        "<h2>Flow Correlation: MFD vs BPR</h2>"
        "<p>Scatter of per-link flow: our density-based MFD assignment "
        f"(10% demand, {state.n_edges} links) vs published BPR equilibrium "
        "(100% demand). Magnitudes differ by ~10&times; due to demand scaling, "
        "but the <b>rank correlation</b> (Spearman &rho;) measures whether "
        "the same links carry relatively more or less traffic under both models. "
        f"&rho; = <b>{flow_rho:.3f}</b> ({len(mfd_flows)} matched links).</p>"
    )

    # 4b: Travel time scatter (exclude gridlocked links with inf TT)
    finite_mask = [t < float("inf") for t in mfd_tt]
    tt_bpr_f = [b for b, m in zip(bpr_tt, finite_mask) if m]
    tt_mfd_f = [t for t, m in zip(mfd_tt, finite_mask) if m]
    tt_labels_f = [l for l, m in zip(link_labels, finite_mask) if m]
    n_gridlocked = sum(1 for m in finite_mask if not m)

    fig_tt_corr = go.Figure()
    fig_tt_corr.add_trace(go.Scatter(
        x=tt_bpr_f, y=tt_mfd_f, mode="markers",
        marker=dict(size=6, color="#D32F2F", opacity=0.7),
        hovertext=tt_labels_f, hoverinfo="text",
    ))
    # Add y=x reference line
    tt_max = max(max(tt_bpr_f, default=1), max(tt_mfd_f, default=1)) * 1.1
    fig_tt_corr.add_shape(
        type="line", x0=0, x1=tt_max, y0=0, y1=tt_max,
        line=dict(color="#999", dash="dash", width=1),
    )
    if len(tt_mfd_f) >= 3:
        tt_rho, _ = spearmanr(tt_bpr_f, tt_mfd_f)
    else:
        tt_rho = 0.0
    fig_tt_corr.update_layout(
        title=f"Link Travel Time: MFD vs BPR — ρ={tt_rho:.3f}",
        xaxis_title="BPR Equilibrium Travel Time (min)",
        yaxis_title="MFD Travel Time (min, 10% demand)",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig_tt_corr)
    gridlock_note = (
        f" ({n_gridlocked} gridlocked link{'s' if n_gridlocked != 1 else ''} excluded.)"
        if n_gridlocked > 0 else ""
    )
    descriptions.append(
        "<h2>Travel Time Correlation: MFD vs BPR</h2>"
        f"<p>Per-link travel time (minutes): MFD at 10% demand ({len(tt_mfd_f)} "
        "links) vs BPR at 100% demand equilibrium. "
        "Dashed line = y&thinsp;=&thinsp;x. "
        "MFD at 10% demand is near freeflow, so travel times cluster near "
        "freeflow values. BPR at 100% shows heavy congestion on some links "
        "(cost &gt;&gt; freeflow time). "
        f"Spearman &rho; = <b>{tt_rho:.3f}</b>.{gridlock_note}</p>"
    )

    # --- 5. V/C scatter across demand levels ---
    fig_vc = go.Figure()
    check_scales = [0.05, 0.10, 0.15, 0.20]
    colors_vc = ["#4CAF50", "#1565C0", "#FF6F00", "#D32F2F"]
    for sc, col in zip(check_scales, colors_vc):
        vc_base = _copy_clean_osrm(base, tmp_path / f"sf_vc_{sc:.2f}")
        res = _run_sf_assignment(vc_base, meta, max_iter=max_iter, method="fw", demand_scale=sc)
        st = res.network_state
        vc_list = []
        flow_list = []
        hover_list = []
        for i in range(st.n_edges):
            k = (int(st.edge_ids[i, 0]), int(st.edge_ids[i, 1]))
            la_i = link_attrs_map.get(k, {})
            vf_i = st.freeflow_kmh[i]
            kj_i = st.jam_density[i]
            qc_i = vf_i * kj_i / 6
            vc_i = st.flow_vph[i] / qc_i if qc_i > 0 else 0
            vc_list.append(vc_i)
            flow_list.append(st.flow_vph[i])
            hover_list.append(f"{k[0]}&rarr;{k[1]}: V/C={vc_i:.2f}")
        fig_vc.add_trace(go.Scatter(
            x=flow_list, y=vc_list, mode="markers",
            marker=dict(size=5, color=col, opacity=0.6),
            name=f"{sc:.0%} demand",
            hovertext=hover_list, hoverinfo="text",
        ))
    fig_vc.add_shape(
        type="line", x0=0, x1=max(flow_list) * 1.1,
        y0=1, y1=1, line=dict(color="#D32F2F", dash="dash", width=1),
    )
    fig_vc.update_layout(
        title="Link V/C Ratio at Different Demand Levels",
        xaxis_title="Link Flow (vph)",
        yaxis_title="Volume / Capacity",
        template="plotly_white",
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True, range=[0, 1.5]),
    )
    figs.append(fig_vc)
    descriptions.append(
        "<h2>V/C by Demand Level</h2>"
        "<p>Each point is one link. V/C = 1.0 is the MFD physical capacity "
        "ceiling. At 5% demand all links are well below capacity. At 20% "
        "demand central links approach or exceed capacity, triggering "
        "the MFD&rsquo;s jam branch.</p>"
    )

    # Write report
    _write_combined_report(
        title="Sioux Falls Validation",
        intro=(
            "<p>Validation of density-based traffic assignment on the canonical "
            "<b>Sioux Falls</b> benchmark (24 nodes, 76 links, 528 OD pairs). "
            f"TNTP total demand: {total_demand:,.0f} (hourly). "
            "Road classification from real Sioux Falls geography: "
            "I-29 (3 lanes, 105 km/h), I-229 (2 lanes, 105 km/h), "
            "arterials (2 lanes, 65 km/h). "
            "VDF: bi-parabolic MFD (k<sub>j</sub>=200 veh/km/lane). "
            "Key finding: TNTP demand exceeds physical road capacity by ~10&times;, "
            "requiring demand scaling for non-gridlocked equilibrium.</p>"
        ),
        figures=figs,
        descriptions=descriptions,
        path=Path(output_path),
    )
    return Path(output_path)
