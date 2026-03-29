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
)
from osrm.assignment.od_matrix import DemandTrip
from osrm.assignment.osm_synthesis import braess_network


def _prepare_network(tmp_path: Path, with_shortcut: bool):
    """Synthesize, extract, partition, customize a Braess network."""
    label = "with" if with_shortcut else "without"
    work = tmp_path / f"braess_{label}"
    work.mkdir()

    osm_path, meta = braess_network(work / "braess.osm", with_shortcut=with_shortcut)
    base = str(work / "braess.osrm")

    osrm.extract(str(osm_path), profile="car", output_path=base, verbosity="ERROR")
    osrm.partition(base, verbosity="ERROR")
    osrm.customize(base, verbosity="ERROR")

    return base, meta


def _run_assignment(base_path: str, meta: dict, demand: float, max_iter: int = 15):
    """Run assignment on Braess network."""
    from osrm.assignment.osm_synthesis import patch_braess_lanes

    trips = [DemandTrip(
        origin=meta["origin"],
        destination=meta["destination"],
        volume=demand,
    )]

    config = AssignmentConfig(
        max_iterations=max_iter,
        convergence_gap=0.0,  # Run all iterations
        smoothing=DensitySmoothingConfig(method="none"),
        verbosity="ERROR",
        speed_csv_dir=str(Path(base_path).parent),
    )

    loop = AssignmentLoop(base_path, config)

    def lane_patch(state):
        patch_braess_lanes(state, meta)

    return loop.run(trips, state_patch=lane_patch)


class TestBraessParadox:
    """Structural validation: Braess paradox.

    The paradox: adding a zero-cost shortcut to a 4-node network
    INCREASES total system travel time under user equilibrium.
    """

    def test_tstt_increases_with_shortcut(self, tmp_path):
        """Core Braess test: TSTT should be higher with the shortcut."""
        base_with, meta_with = _prepare_network(tmp_path, with_shortcut=True)
        base_without, meta_without = _prepare_network(tmp_path, with_shortcut=False)

        result_with = _run_assignment(base_with, meta_with, demand=3000.0)
        result_without = _run_assignment(base_without, meta_without, demand=3000.0)

        tstt_with = result_with.iteration_log[-1].tstt
        tstt_without = result_without.iteration_log[-1].tstt

        assert tstt_with > tstt_without, (
            f"Braess paradox not observed: TSTT with shortcut ({tstt_with:.0f}) "
            f"should exceed TSTT without ({tstt_without:.0f})"
        )

    def test_both_complete_without_error(self, tmp_path):
        """Both networks should complete assignment without error."""
        base_with, meta_with = _prepare_network(tmp_path, with_shortcut=True)
        base_without, meta_without = _prepare_network(tmp_path, with_shortcut=False)

        result_with = _run_assignment(base_with, meta_with, demand=2000.0, max_iter=5)
        result_without = _run_assignment(base_without, meta_without, demand=2000.0, max_iter=5)

        assert result_with.iterations == 5
        assert result_without.iterations == 5
        assert result_with.network_state.n_edges > 0
        assert result_without.network_state.n_edges > 0

    def test_flow_nonnegativity(self, tmp_path):
        """All link flows must be non-negative."""
        base, meta = _prepare_network(tmp_path, with_shortcut=True)
        result = _run_assignment(base, meta, demand=3000.0)
        assert np.all(result.network_state.flow_vph >= 0)

    def test_speeds_within_bounds(self, tmp_path):
        """Speeds must be between VDF min_speed and freeflow."""
        base, meta = _prepare_network(tmp_path, with_shortcut=True)
        result = _run_assignment(base, meta, demand=3000.0)
        state = result.network_state
        assert np.all(state.speed_kmh >= 0.01 - 1e-6)
        assert np.all(state.speed_kmh <= state.freeflow_kmh + 1e-6)


def generate_braess_report(
    tmp_path: str | Path,
    output_path: str = "docs/plots/braess_validation.html",
    demand: float = 3000.0,
    max_iter: int = 100,
) -> Path:
    """Run Braess validation and generate an interactive HTML report.

    Parameters
    ----------
    tmp_path : str or Path
        Working directory for temporary OSRM files.
    output_path : str
        Where to write the HTML report.
    demand : float
        Demand volume (vehicles per period).
    max_iter : int
        Assignment iterations.

    Returns
    -------
    Path to the generated report.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from osrm.assignment.plots import _write_combined_report

    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Run both scenarios
    base_with, meta_with = _prepare_network(tmp_path, with_shortcut=True)
    base_without, meta_without = _prepare_network(tmp_path, with_shortcut=False)

    result_with = _run_assignment(base_with, meta_with, demand, max_iter)
    result_without = _run_assignment(base_without, meta_without, demand, max_iter)

    figs = []
    descriptions = []

    # --- 0. Network topology diagram ---
    node_pos = {1: (0, 0.5), 3: (0.5, 1), 4: (0.5, 0), 2: (1, 0.5)}
    edges_wo = [(1, 3), (1, 4), (3, 2), (4, 2)]
    edges_w = edges_wo + [(3, 4)]
    edge_styles = {
        (1, 3): ("narrow 1-lane 60 km/h", "#F44336"),
        (1, 4): ("wide 3-lane 40 km/h", "#2196F3"),
        (3, 2): ("wide 3-lane 40 km/h", "#2196F3"),
        (4, 2): ("narrow 1-lane 60 km/h", "#F44336"),
        (3, 4): ("shortcut 1-lane 60 km/h", "#FF9800"),
    }

    topo_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Without Shortcut", "With Shortcut"],
        horizontal_spacing=0.12,
    )

    for col, edges in enumerate([edges_wo, edges_w], 1):
        for u, v in edges:
            x0, y0 = node_pos[u]
            x1, y1 = node_pos[v]
            _, color = edge_styles[(u, v)]
            topo_fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="text",
                hovertext=f"{u}→{v}: {edge_styles[(u,v)][0]}",
                showlegend=False,
            ), row=1, col=col)
            topo_fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref=f"x{col}" if col > 1 else "x",
                yref=f"y{col}" if col > 1 else "y",
                axref=f"x{col}" if col > 1 else "x",
                ayref=f"y{col}" if col > 1 else "y",
                showarrow=True, arrowhead=3, arrowsize=1.5,
                arrowcolor=color, arrowwidth=2,
            )
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = y1 - y0, -(x1 - x0)
            mag = max((dx**2 + dy**2) ** 0.5, 1e-9)
            ox, oy = 0.06 * dx / mag, 0.06 * dy / mag
            topo_fig.add_annotation(
                x=mx + ox, y=my + oy, text=f"{u}→{v}",
                xref=f"x{col}" if col > 1 else "x",
                yref=f"y{col}" if col > 1 else "y",
                showarrow=False, font=dict(size=10, color=color),
            )

        xs = [node_pos[n][0] for n in sorted(node_pos)]
        ys = [node_pos[n][1] for n in sorted(node_pos)]
        labels = [str(n) for n in sorted(node_pos)]
        roles = {1: "Origin", 2: "Destination", 3: "Node 3", 4: "Node 4"}
        hover = [f"Node {n} ({roles[n]})" for n in sorted(node_pos)]
        colors = ["#4CAF50" if n == 1 else "#F44336" if n == 2 else "#9E9E9E"
                  for n in sorted(node_pos)]
        topo_fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=28, color=colors, line=dict(width=2, color="white")),
            text=labels, textfont=dict(size=14, color="white"),
            textposition="middle center",
            hovertext=hover, hoverinfo="text",
            showlegend=False,
        ), row=1, col=col)

    for suffix in ["", "2"]:
        xref, yref = f"xaxis{suffix}", f"yaxis{suffix}"
        topo_fig.update_layout(**{
            xref: dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-0.15, 1.15], fixedrange=True),
            yref: dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-0.15, 1.15], scaleanchor=f"x{suffix}" if suffix else "x",
                       fixedrange=True),
        })
    topo_fig.update_layout(
        title="Network Topology",
        template="plotly_white",
        height=350,
    )

    figs.append(topo_fig)
    descriptions.append("""<h2>Network Topology</h2>
    <p>The Braess diamond network: <span style="color:#4CAF50">●</span> Origin (node 1),
    <span style="color:#F44336">●</span> Destination (node 2).
    <span style="color:#F44336">Red</span> = narrow (1-lane, 60 km/h — congestion-sensitive).
    <span style="color:#2196F3">Blue</span> = wide (3-lane, 40 km/h — high capacity).
    <span style="color:#FF9800">Orange</span> = shortcut (1-lane, 60 km/h).
    All arterials are ~4 km; the shortcut is ~0.6 km.</p>""")

    # --- 1. Link state comparison table ---
    def _state_table(result_w, result_wo):
        """Build an HTML table comparing link density, speed, and flow."""
        from collections import OrderedDict
        rows = []
        for result, scenario in [(result_w, "With"), (result_wo, "Without")]:
            state = result.network_state
            for i in range(state.n_edges):
                label = f"{int(state.edge_ids[i,0])}→{int(state.edge_ids[i,1])}"
                rows.append((
                    label, scenario,
                    state.density_vpkm[i],
                    state.speed_kmh[i],
                    state.freeflow_kmh[i],
                    state.flow_vph[i],
                    state.jam_density[i],
                    state.n_lanes[i],
                ))

        pivot: dict[str, dict[str, tuple]] = OrderedDict()
        for label, scenario, k, v, vf, q, kj, lanes in rows:
            pivot.setdefault(label, {"lanes": lanes, "kj": kj, "vf": vf})[scenario] = (k, v, q)

        html = (
            '<table style="border-collapse:collapse; width:100%; max-width:900px; '
            'margin:12px auto; font-family:system-ui,sans-serif; font-size:0.9em;">'
            '<thead><tr style="border-bottom:2px solid #333;">'
            '<th style="text-align:left;padding:8px;">Link</th>'
            '<th style="padding:8px;">Lanes</th>'
            '<th style="padding:8px;">k<sub>j</sub></th>'
            '<th style="padding:8px;">v<sub>f</sub></th>'
            '<th colspan="3" style="text-align:center;padding:8px;border-left:2px solid #ccc;">With Shortcut</th>'
            '<th colspan="3" style="text-align:center;padding:8px;border-left:2px solid #ccc;">Without Shortcut</th>'
            '</tr><tr style="border-bottom:1px solid #999;">'
            '<th></th><th></th><th></th><th></th>'
            '<th style="padding:4px 8px;border-left:2px solid #ccc;">k</th>'
            '<th style="padding:4px 8px;">v</th>'
            '<th style="padding:4px 8px;">q</th>'
            '<th style="padding:4px 8px;border-left:2px solid #ccc;">k</th>'
            '<th style="padding:4px 8px;">v</th>'
            '<th style="padding:4px 8px;">q</th>'
            '</tr></thead><tbody>'
        )

        def _v_color(v, vf):
            ratio = v / vf if vf > 0 else 1
            if ratio < 0.1:
                return "#F44336"
            elif ratio < 0.5:
                return "#FF9800"
            return "#4CAF50"

        def _k_style(k, kj):
            ratio = k / kj if kj > 0 else 0
            if ratio > 0.9:
                return "font-weight:700;color:#F44336;"
            elif ratio > 0.5:
                return "color:#FF9800;"
            return ""

        for link, info in pivot.items():
            lanes = info["lanes"]
            kj = info["kj"]
            vf = info["vf"]
            w = info.get("With")
            wo = info.get("Without")

            def _cells(vals):
                if vals is None:
                    return '<td style="text-align:right;padding:4px 8px;border-left:2px solid #ccc;">—</td>' \
                           '<td style="text-align:right;padding:4px 8px;">—</td>' \
                           '<td style="text-align:right;padding:4px 8px;">—</td>'
                k, v, q = vals
                return (
                    f'<td style="text-align:right;padding:4px 8px;border-left:2px solid #ccc;{_k_style(k, kj)}">{k:.1f}</td>'
                    f'<td style="text-align:right;padding:4px 8px;color:{_v_color(v, vf)};">{v:.2f}</td>'
                    f'<td style="text-align:right;padding:4px 8px;">{q:.0f}</td>'
                )

            html += (
                f'<tr style="border-bottom:1px solid #e0e0e0;">'
                f'<td style="padding:4px 8px;font-weight:600;">{link}</td>'
                f'<td style="text-align:center;padding:4px 8px;">{lanes}</td>'
                f'<td style="text-align:center;padding:4px 8px;">{kj:.0f}</td>'
                f'<td style="text-align:center;padding:4px 8px;">{vf:.0f}</td>'
                f'{_cells(w)}{_cells(wo)}'
                f'</tr>'
            )
        html += '</tbody></table>'
        return html

    figs.append(None)
    descriptions.append(
        """<h2>Link State at Equilibrium</h2>
        <p>Density (k, veh/km), speed (v, km/h), and flow (q = k×v, veh/hr) at the
        final iteration. <span style="color:#F44336;font-weight:600;">Red density</span>
        = near jam (k/k<sub>j</sub> &gt; 0.9).
        <span style="color:#F44336">Red speed</span> = severe congestion (v/v<sub>f</sub> &lt; 0.1).
        <span style="color:#FF9800">Orange</span> = moderate.
        <span style="color:#4CAF50">Green</span> = uncongested.</p>"""
        + _state_table(result_with, result_without)
    )

    # --- 2. TSTT Comparison ---
    tstt_with = [r.tstt for r in result_with.iteration_log]
    tstt_without = [r.tstt for r in result_without.iteration_log]
    iters_w = [r.iteration for r in result_with.iteration_log]
    iters_wo = [r.iteration for r in result_without.iteration_log]

    fig_tstt = go.Figure()
    fig_tstt.add_trace(go.Scatter(
        x=iters_w, y=tstt_with, mode="lines+markers",
        name="With shortcut", line=dict(color="#F44336", width=2),
    ))
    fig_tstt.add_trace(go.Scatter(
        x=iters_wo, y=tstt_without, mode="lines+markers",
        name="Without shortcut", line=dict(color="#2196F3", width=2),
    ))
    fig_tstt.update_layout(
        title="Total System Travel Time (TSTT) per Iteration",
        xaxis_title="Iteration", yaxis_title="TSTT (veh·seconds)",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig_tstt)

    delta = tstt_with[-1] - tstt_without[-1]
    pct = delta / tstt_without[-1] * 100 if tstt_without[-1] > 0 else 0
    gap_w = result_with.iteration_log[-1].relative_gap
    gap_wo = result_without.iteration_log[-1].relative_gap
    descriptions.append(f"""<h2>TSTT Convergence</h2>
    <p>Red = network WITH shortcut, blue = WITHOUT.
    TSTT with shortcut = <b>{tstt_with[-1]:,.0f}</b> veh·s (gap={gap_w:.6f}),
    without = <b>{tstt_without[-1]:,.0f}</b> veh·s (gap={gap_wo:.6f}).
    Δ = <b>{delta:+,.0f}</b> ({pct:+.1f}%).
    {"✅ <b>Braess paradox confirmed</b>: adding the shortcut <i>increases</i> total travel time."
     if delta > 0 else "⚠️ Paradox not observed at this demand level."}</p>""")

    # --- 3. Convergence (gap) ---
    gap_with = [r.relative_gap for r in result_with.iteration_log]
    gap_without = [r.relative_gap for r in result_without.iteration_log]

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(
        x=iters_w, y=gap_with, mode="lines+markers",
        name="With shortcut", line=dict(color="#F44336", width=2),
    ))
    fig_gap.add_trace(go.Scatter(
        x=iters_wo, y=gap_without, mode="lines+markers",
        name="Without shortcut", line=dict(color="#2196F3", width=2),
    ))
    fig_gap.update_layout(
        title="Wardrop Relative Gap per Iteration",
        xaxis_title="Iteration", yaxis_title="Relative Gap",
        yaxis_type="log",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig_gap)
    descriptions.append("""<h2>Convergence</h2>
    <p>Relative gap measures proximity to Wardrop user equilibrium (gap = 0 means
    all used paths have equal cost). Log scale. The symmetric "without" scenario
    converges faster; the asymmetric 3-path "with" scenario is slower but
    steadily decreasing under MSA (α = 1/n).</p>""")

    # Write report
    _write_combined_report(
        title="Braess Paradox Validation",
        intro=f"""<p>Structural validation of the density-based traffic assignment using the
        <b>Braess paradox</b> — a 4-node diamond network where adding a shortcut link
        increases total system travel time under user equilibrium.</p>
        <p>Demand: <b>{demand:,.0f}</b> vehicles.
        <b>{max_iter}</b> MSA iterations per scenario.
        VDF: bi-parabolic speed-density (Fournier et al.), k<sub>c</sub> = k<sub>j</sub>/3.</p>""",
        figures=figs,
        descriptions=descriptions,
        path=Path(output_path),
    )
    return Path(output_path)
