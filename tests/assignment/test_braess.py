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
    return loop.run(trips)


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
        """Speeds must be between min_speed and freeflow."""
        base, meta = _prepare_network(tmp_path, with_shortcut=True)
        result = _run_assignment(base, meta, demand=3000.0)
        state = result.network_state
        assert np.all(state.speed_kmh >= 5.0 - 1e-6)
        assert np.all(state.speed_kmh <= state.freeflow_kmh + 1e-6)


def generate_braess_report(
    tmp_path: str | Path,
    output_path: str = "docs/plots/braess_validation.html",
    demand: float = 3000.0,
    max_iter: int = 15,
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

    # --- 1. TSTT Comparison ---
    tstt_with = [r.tstt for r in result_with.iteration_log]
    tstt_without = [r.tstt for r in result_without.iteration_log]
    iters_w = [r.iteration for r in result_with.iteration_log]
    iters_wo = [r.iteration for r in result_without.iteration_log]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=iters_w, y=tstt_with, mode="lines+markers",
        name="With shortcut", line=dict(color="#F44336", width=2),
    ))
    fig1.add_trace(go.Scatter(
        x=iters_wo, y=tstt_without, mode="lines+markers",
        name="Without shortcut", line=dict(color="#2196F3", width=2),
    ))
    fig1.update_layout(
        title="Total System Travel Time (TSTT) per Iteration",
        xaxis_title="Iteration", yaxis_title="TSTT (veh·seconds)",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig1)

    delta = tstt_with[-1] - tstt_without[-1]
    pct = delta / tstt_without[-1] * 100 if tstt_without[-1] > 0 else 0
    descriptions.append(f"""<h2>1. Braess Paradox: TSTT Comparison</h2>
    <p>The <b>Braess paradox</b> states that adding a link to a network can <i>increase</i>
    total system travel time under user equilibrium. Red = network WITH shortcut,
    blue = WITHOUT.</p>
    <p><b>Result</b>: TSTT with shortcut = <b>{tstt_with[-1]:,.0f}</b> veh·s,
    without = <b>{tstt_without[-1]:,.0f}</b> veh·s.
    Δ = <b>{delta:+,.0f}</b> ({pct:+.1f}%).
    {"✅ Paradox confirmed!" if delta > 0 else "⚠️ Paradox NOT observed."}</p>""")

    # --- 2. Convergence Comparison ---
    gap_with = [r.relative_gap for r in result_with.iteration_log]
    gap_without = [r.relative_gap for r in result_without.iteration_log]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=iters_w, y=gap_with, mode="lines+markers",
        name="With shortcut", line=dict(color="#F44336", width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=iters_wo, y=gap_without, mode="lines+markers",
        name="Without shortcut", line=dict(color="#2196F3", width=2),
    ))
    fig2.update_layout(
        title="Relative Gap per Iteration",
        xaxis_title="Iteration", yaxis_title="Relative Gap",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig2)
    descriptions.append("""<h2>2. Convergence</h2>
    <p>Relative gap measures distance from Wardrop user equilibrium. A gap of 0
    means all used paths have equal cost. Both scenarios should converge toward zero.</p>""")

    # --- 3. Link Flow Comparison (table) ---
    def _flow_table(result_w, result_wo):
        """Build an HTML table comparing link flows between scenarios."""
        rows = []
        # Collect all edge labels from both scenarios
        for result, scenario in [(result_w, "With"), (result_wo, "Without")]:
            state = result.network_state
            for i in range(state.n_edges):
                label = f"{int(state.edge_ids[i,0])}→{int(state.edge_ids[i,1])}"
                rows.append((label, scenario, state.flow_vph[i]))

        # Pivot into {link: {With: flow, Without: flow}}
        from collections import OrderedDict
        pivot: dict[str, dict[str, float]] = OrderedDict()
        for label, scenario, flow in rows:
            pivot.setdefault(label, {})[scenario] = flow

        html = (
            '<table style="border-collapse:collapse; width:100%; max-width:600px; '
            'margin:12px auto; font-family:system-ui,sans-serif;">'
            '<thead><tr style="border-bottom:2px solid #333;">'
            '<th style="text-align:left;padding:8px;">Link</th>'
            '<th style="text-align:right;padding:8px;">With Shortcut (veh/hr)</th>'
            '<th style="text-align:right;padding:8px;">Without Shortcut (veh/hr)</th>'
            '</tr></thead><tbody>'
        )
        for link, vals in pivot.items():
            fw = vals.get("With", 0)
            fwo = vals.get("Without", 0)
            html += (
                f'<tr style="border-bottom:1px solid #e0e0e0;">'
                f'<td style="padding:6px 8px;font-weight:600;">{link}</td>'
                f'<td style="text-align:right;padding:6px 8px;">{fw:,.1f}</td>'
                f'<td style="text-align:right;padding:6px 8px;">{fwo:,.1f}</td>'
                f'</tr>'
            )
        html += '</tbody></table>'
        return html

    figs.append(None)
    descriptions.append(
        """<h2>3. Link Flows at Equilibrium</h2>
        <p>Per-link flow at the final iteration. In the classic Braess network, the shortcut
        causes traffic to concentrate on fewer links, overloading them.</p>"""
        + _flow_table(result_with, result_without)
    )

    # --- 4. Speed Reduction (table) ---
    def _speed_table(result_w, result_wo):
        """Build an HTML table comparing speed ratios between scenarios."""
        rows = []
        for result, scenario in [(result_w, "With"), (result_wo, "Without")]:
            state = result.network_state
            ratio = state.speed_kmh / np.maximum(state.freeflow_kmh, 1.0)
            for i in range(state.n_edges):
                label = f"{int(state.edge_ids[i,0])}→{int(state.edge_ids[i,1])}"
                rows.append((label, scenario, ratio[i], state.speed_kmh[i], state.freeflow_kmh[i]))

        from collections import OrderedDict
        pivot: dict[str, dict[str, tuple]] = OrderedDict()
        for label, scenario, r, spd, ff in rows:
            pivot.setdefault(label, {})[scenario] = (r, spd, ff)

        html = (
            '<table style="border-collapse:collapse; width:100%; max-width:700px; '
            'margin:12px auto; font-family:system-ui,sans-serif;">'
            '<thead><tr style="border-bottom:2px solid #333;">'
            '<th style="text-align:left;padding:8px;">Link</th>'
            '<th style="text-align:left;padding:8px;">v<sub>f</sub> (km/h)</th>'
            '<th style="text-align:right;padding:8px;">With Shortcut</th>'
            '<th style="text-align:right;padding:8px;">Without Shortcut</th>'
            '</tr></thead><tbody>'
        )

        def _color(r):
            if r < 0.8:
                return "#F44336"
            elif r < 0.95:
                return "#FF9800"
            return "#4CAF50"

        for link, vals in pivot.items():
            rw, sw, ffw = vals.get("With", (1.0, 0, 0))
            rwo, swo, ffwo = vals.get("Without", (1.0, 0, 0))
            ff = ffw or ffwo
            html += (
                f'<tr style="border-bottom:1px solid #e0e0e0;">'
                f'<td style="padding:6px 8px;font-weight:600;">{link}</td>'
                f'<td style="padding:6px 8px;">{ff:.0f}</td>'
                f'<td style="text-align:right;padding:6px 8px;">'
                f'<span style="color:{_color(rw)};font-weight:600;">{rw:.2f}</span>'
                f' ({sw:.1f} km/h)</td>'
                f'<td style="text-align:right;padding:6px 8px;">'
                f'<span style="color:{_color(rwo)};font-weight:600;">{rwo:.2f}</span>'
                f' ({swo:.1f} km/h)</td>'
                f'</tr>'
            )
        html += '</tbody></table>'
        return html

    figs.append(None)
    descriptions.append(
        """<h2>4. Speed Reduction by Link</h2>
        <p>Ratio of equilibrium speed to free-flow speed (v/v<sub>f</sub>).
        <span style="color:#4CAF50;font-weight:600;">Green</span> ≥ 0.95 (uncongested),
        <span style="color:#FF9800;font-weight:600;">orange</span> 0.8–0.95 (moderate),
        <span style="color:#F44336;font-weight:600;">red</span> &lt; 0.8 (congested).</p>"""
        + _speed_table(result_with, result_without)
    )

    # --- 0. Network topology diagram (prepend) ---
    node_pos = {1: (0, 0.5), 3: (0.5, 1), 4: (0.5, 0), 2: (1, 0.5)}
    edges_wo = [(1, 3), (1, 4), (3, 2), (4, 2)]
    edges_w = edges_wo + [(3, 4)]
    edge_styles = {
        (1, 3): ("secondary 1-lane 50km/h", "#F44336"),
        (1, 4): ("primary 4-lane 30km/h", "#2196F3"),
        (3, 2): ("primary 4-lane 30km/h", "#2196F3"),
        (4, 2): ("secondary 1-lane 50km/h", "#F44336"),
        (3, 4): ("shortcut 4-lane 60km/h", "#FF9800"),
    }

    topo_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Without Shortcut", "With Shortcut"],
        horizontal_spacing=0.12,
    )

    for col, edges in enumerate([edges_wo, edges_w], 1):
        # Draw edges as arrows
        for u, v in edges:
            x0, y0 = node_pos[u]
            x1, y1 = node_pos[v]
            _, color = edge_styles[(u, v)]
            # Line for the edge
            topo_fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="text",
                hovertext=f"{u}→{v}: {edge_styles[(u,v)][0]}",
                showlegend=False,
            ), row=1, col=col)
            # Arrowhead via annotation
            topo_fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref=f"x{col}" if col > 1 else "x",
                yref=f"y{col}" if col > 1 else "y",
                axref=f"x{col}" if col > 1 else "x",
                ayref=f"y{col}" if col > 1 else "y",
                showarrow=True, arrowhead=3, arrowsize=1.5,
                arrowcolor=color, arrowwidth=2,
            )
            # Edge label
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            # Offset label slightly so it doesn't overlap the line
            dx, dy = y1 - y0, -(x1 - x0)
            mag = max((dx**2 + dy**2) ** 0.5, 1e-9)
            ox, oy = 0.06 * dx / mag, 0.06 * dy / mag
            topo_fig.add_annotation(
                x=mx + ox, y=my + oy, text=f"{u}→{v}",
                xref=f"x{col}" if col > 1 else "x",
                yref=f"y{col}" if col > 1 else "y",
                showarrow=False, font=dict(size=10, color=color),
            )

        # Draw nodes
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

    # Prepend topology as first figure
    figs.insert(0, topo_fig)
    descriptions.insert(0, """<h2>Network Topology</h2>
    <p>The Braess diamond network: <span style="color:#4CAF50">●</span> Origin (node 1),
    <span style="color:#F44336">●</span> Destination (node 2).
    <span style="color:#F44336">Red</span> links are narrow (1-lane, 50 km/h — congestion-sensitive).
    <span style="color:#2196F3">Blue</span> links are wide (4-lane, 30 km/h — effectively constant cost).
    <span style="color:#FF9800">Orange</span> is the shortcut (4-lane, 60 km/h).
    The paradox: adding the shortcut <i>increases</i> total system travel time.</p>""")

    # Write report
    _write_combined_report(
        title="Braess Paradox Validation",
        intro=f"""<p>Structural validation of the traffic assignment using the <b>Braess paradox</b> —
        a 4-node diamond network where adding a shortcut link increases total system travel time.
        Demand: {demand:.0f} vehicles. {max_iter} MSA iterations per scenario.</p>""",
        figures=figs,
        descriptions=descriptions,
        path=Path(output_path),
    )
    return Path(output_path)
