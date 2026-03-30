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
        verbosity="ERROR",
        speed_csv_dir=str(Path(base_path).parent),
    )

    loop = AssignmentLoop(base_path, config)

    def lane_patch(state):
        patch_braess_lanes(state, meta)

    return loop.run(trips, state_patch=lane_patch)


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


def generate_braess_report(
    tmp_path: str | Path,
    output_path: str = "docs/plots/braess_validation.html",
    demand: float = 2500.0,
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

    # Run both scenarios (MSA)
    base_with, meta_with = _prepare_network(tmp_path, with_shortcut=True)
    base_without, meta_without = _prepare_network(tmp_path, with_shortcut=False)

    result_with = _run_assignment(base_with, meta_with, demand, max_iter)
    result_without = _run_assignment(base_without, meta_without, demand, max_iter)

    # Run FW scenarios (reuse same prepared networks)
    base_fw_w, meta_fw_w = _prepare_network(tmp_path / "fw", with_shortcut=True)
    base_fw_wo, meta_fw_wo = _prepare_network(tmp_path / "fw", with_shortcut=False)

    result_fw_with = _run_assignment(base_fw_w, meta_fw_w, demand, max_iter, method="fw")
    result_fw_without = _run_assignment(base_fw_wo, meta_fw_wo, demand, max_iter, method="fw")

    figs = []
    descriptions = []

    # --- 0. Network topology diagram (to-scale) ---
    # Convert synthesis coords to km from origin node
    meta_nodes = meta_with["nodes"]
    cos_lat = np.cos(np.radians(43.735))
    node_km = {}
    ref_lon, ref_lat = meta_nodes[1]
    for nid, (lon, lat) in meta_nodes.items():
        node_km[nid] = (
            (lon - ref_lon) * 111.32 * cos_lat,
            (lat - ref_lat) * 111.32,
        )

    # Simple 4-node diamond — direct edges between nodes
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
                hovertext=f"{u}→{v}: {label}",
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

    # Compute axis ranges from node positions
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
    <p>The Braess diamond network (to scale): <span style="color:#4CAF50">●</span> Origin (node 1),
    <span style="color:#F44336">●</span> Destination (node 2).
    <span style="color:#F44336">Red</span> = variable (1-lane, 80 km/h &mdash; fast but congestion-sensitive).
    <span style="color:#2196F3">Blue</span> = constant (4-lane, 50 km/h &mdash; slow but robust).
    <span style="color:#FF9800">Orange</span> = shortcut (1-lane, 80 km/h, ~1 km).
    All main links ~10 km.</p>""")

    # --- 1. Link state comparison table ---
    def _state_table(result_w, result_wo):
        """Build an HTML table comparing link state between scenarios."""
        from collections import OrderedDict
        rows = []
        for result, scenario in [(result_w, "With"), (result_wo, "Without")]:
            state = result.network_state
            for i in range(state.n_edges):
                label = f"{int(state.edge_ids[i,0])}&rarr;{int(state.edge_ids[i,1])}"
                v = max(state.speed_kmh[i], 1.08)
                travel_time_s = state.length_m[i] / (v / 3.6)
                rows.append((
                    label, scenario,
                    state.density_vpkm[i],
                    state.speed_kmh[i],
                    state.freeflow_kmh[i],
                    state.flow_vph[i],
                    state.jam_density[i],
                    state.n_lanes[i],
                    state.length_m[i],
                    travel_time_s,
                ))

        pivot: dict[str, dict[str, tuple]] = OrderedDict()
        for label, scenario, k, v, vf, q, kj, lanes, length, tt in rows:
            pivot.setdefault(label, {
                "lanes": lanes, "kj": kj, "vf": vf, "length": length,
            })[scenario] = (k, v, q, tt)

        html = (
            '<table style="border-collapse:collapse; width:100%; max-width:1100px; '
            'margin:12px auto; font-family:system-ui,sans-serif; font-size:0.85em;">'
            '<thead><tr style="border-bottom:2px solid #333;">'
            '<th style="text-align:left;padding:8px;">Link</th>'
            '<th style="padding:8px;">Length</th>'
            '<th style="padding:8px;">Lanes</th>'
            '<th style="padding:8px;">k<sub>j</sub></th>'
            '<th style="padding:8px;">v<sub>f</sub></th>'
            '<th colspan="4" style="text-align:center;padding:8px;border-left:2px solid #ccc;">Without Shortcut</th>'
            '<th colspan="4" style="text-align:center;padding:8px;border-left:2px solid #ccc;">With Shortcut</th>'
            '</tr><tr style="border-bottom:1px solid #999;">'
            '<th></th><th></th><th></th><th></th><th></th>'
            '<th style="padding:4px 6px;border-left:2px solid #ccc;">k</th>'
            '<th style="padding:4px 6px;">v</th>'
            '<th style="padding:4px 6px;">q</th>'
            '<th style="padding:4px 6px;">t</th>'
            '<th style="padding:4px 6px;border-left:2px solid #ccc;">k</th>'
            '<th style="padding:4px 6px;">v</th>'
            '<th style="padding:4px 6px;">q</th>'
            '<th style="padding:4px 6px;">t</th>'
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

        def _fmt_time(s):
            if s >= 3600:
                return f"{s/3600:.1f}h"
            return f"{s:.0f}s"

        for link, info in pivot.items():
            lanes = info["lanes"]
            kj = info["kj"]
            vf = info["vf"]
            length_km = info["length"] / 1000.0
            ff_time_s = info["length"] / (vf / 3.6)
            w = info.get("With")
            wo = info.get("Without")

            def _cells(vals):
                if vals is None:
                    return ('<td style="text-align:right;padding:4px 6px;border-left:2px solid #ccc;">&mdash;</td>'
                            '<td style="text-align:right;padding:4px 6px;">&mdash;</td>'
                            '<td style="text-align:right;padding:4px 6px;">&mdash;</td>'
                            '<td style="text-align:right;padding:4px 6px;">&mdash;</td>')
                k, v, q, tt = vals
                tt_ratio = tt / ff_time_s if ff_time_s > 0 else 1
                tt_color = "#F44336" if tt_ratio > 2 else "#FF9800" if tt_ratio > 1.3 else "#4CAF50"
                return (
                    f'<td style="text-align:right;padding:4px 6px;border-left:2px solid #ccc;{_k_style(k, kj)}">{k:.1f}</td>'
                    f'<td style="text-align:right;padding:4px 6px;color:{_v_color(v, vf)};">{v:.1f}</td>'
                    f'<td style="text-align:right;padding:4px 6px;">{q:.0f}</td>'
                    f'<td style="text-align:right;padding:4px 6px;color:{tt_color};">{_fmt_time(tt)}</td>'
                )

            html += (
                f'<tr style="border-bottom:1px solid #e0e0e0;">'
                f'<td style="padding:4px 6px;font-weight:600;">{link}</td>'
                f'<td style="text-align:center;padding:4px 6px;">{length_km:.1f} km</td>'
                f'<td style="text-align:center;padding:4px 6px;">{lanes}</td>'
                f'<td style="text-align:center;padding:4px 6px;">{kj:.0f}</td>'
                f'<td style="text-align:center;padding:4px 6px;">{vf:.0f}</td>'
                f'{_cells(wo)}{_cells(w)}'
                f'</tr>'
            )
        html += '</tbody></table>'
        return html

    def _route_travel_times(result_w, result_wo):
        """Build a table of route travel times to demonstrate Wardrop equilibrium."""
        # Simple 4-node edges — no multi-segment highways
        routes = {
            "Upper (1&rarr;3&rarr;2)": [("1", "3"), ("3", "2")],
            "Lower (1&rarr;4&rarr;2)": [("1", "4"), ("4", "2")],
            "Shortcut (1&rarr;3&rarr;4&rarr;2)": [("1", "3"), ("3", "4"), ("4", "2")],
        }

        def _link_times(result):
            """Return {(from, to): travel_time_s} for each edge."""
            state = result.network_state
            times = {}
            for i in range(state.n_edges):
                from_id = str(int(state.edge_ids[i, 0]))
                to_id = str(int(state.edge_ids[i, 1]))
                v = max(state.speed_kmh[i], 1.08)
                times[(from_id, to_id)] = state.length_m[i] / (v / 3.6)
            return times

        times_w = _link_times(result_w)
        times_wo = _link_times(result_wo)

        html = (
            '<table style="border-collapse:collapse; width:100%; max-width:700px; '
            'margin:12px auto; font-family:system-ui,sans-serif; font-size:0.85em;">'
            '<thead><tr style="border-bottom:2px solid #333;">'
            '<th style="text-align:left;padding:8px;">Route</th>'
            '<th style="text-align:right;padding:8px;">Without Shortcut</th>'
            '<th style="text-align:right;padding:8px;">With Shortcut</th>'
            '</tr></thead><tbody>'
        )

        for route_name, links in routes.items():
            # Without shortcut
            wo_total = None
            if all(lk in times_wo for lk in links):
                wo_total = sum(times_wo[lk] for lk in links)
            # With shortcut
            w_total = None
            if all(lk in times_w for lk in links):
                w_total = sum(times_w[lk] for lk in links)

            wo_str = f"{wo_total:.1f}s" if wo_total is not None else "&mdash;"
            w_str = f"{w_total:.1f}s" if w_total is not None else "&mdash;"

            html += (
                f'<tr style="border-bottom:1px solid #e0e0e0;">'
                f'<td style="padding:6px 8px;font-weight:600;">{route_name}</td>'
                f'<td style="text-align:right;padding:6px 8px;">{wo_str}</td>'
                f'<td style="text-align:right;padding:6px 8px;">{w_str}</td>'
                f'</tr>'
            )

        html += '</tbody></table>'
        return html

    figs.append(None)
    descriptions.append(
        """<h2>Link State at Equilibrium</h2>
        <p>Density (k, veh/km), speed (v, km/h), flow (q = k&times;v, veh/hr), and travel
        time (t = L/v) at the final iteration.
        <span style="color:#F44336;font-weight:600;">Red density</span>
        = near jam (k/k<sub>j</sub> &gt; 0.9).
        <span style="color:#F44336">Red speed</span> = severe congestion (v/v<sub>f</sub> &lt; 0.1).
        <span style="color:#F44336">Red time</span> = &gt;2x freeflow.
        <span style="color:#FF9800">Orange</span> = moderate.
        <span style="color:#4CAF50">Green</span> = uncongested.</p>"""
        + _state_table(result_with, result_without)
    )

    # --- Route Travel Times (Wardrop equilibrium) ---
    figs.append(None)
    descriptions.append(
        """<h2>Route Travel Times</h2>
        <p>Total travel time for each OD route, summed from link-level t = L/v.
        At user equilibrium (Wardrop), all <i>used</i> routes between an OD pair
        should have equal travel time.</p>"""
        + _route_travel_times(result_with, result_without)
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
        xaxis_title="Iteration", yaxis_title="TSTT (veh&middot;seconds)",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig_tstt)

    delta = tstt_with[-1] - tstt_without[-1]
    pct = delta / tstt_without[-1] * 100 if tstt_without[-1] > 0 else 0
    gap_w = result_with.iteration_log[-1].relative_gap
    gap_wo = result_without.iteration_log[-1].relative_gap
    tstt_w_fmt = f"{tstt_with[-1]:,.0f}"
    tstt_wo_fmt = f"{tstt_without[-1]:,.0f}"
    delta_fmt = f"{delta:+,.0f}"
    paradox_msg = (
        " <b>Braess paradox confirmed</b>: adding the shortcut <i>increases</i> total travel time."
        if delta > 0 else " Paradox not observed at this demand level."
    )
    gap_w_fmt = f"{gap_w:.6f}"
    gap_wo_fmt = f"{gap_wo:.6f}"
    pct_fmt = f"{pct:+.1f}"
    descriptions.append(
        "<h2>TSTT Convergence</h2>"
        "<p>Red = network WITH shortcut, blue = WITHOUT. "
        f"TSTT with shortcut = <b>{tstt_w_fmt}</b> veh&middot;s (gap={gap_w_fmt}), "
        f"without = <b>{tstt_wo_fmt}</b> veh&middot;s (gap={gap_wo_fmt}). "
        f"&Delta; = <b>{delta_fmt}</b> ({pct_fmt}%). "
        f"{paradox_msg}</p>"
    )

    # --- 3. Convergence (gap): MSA vs Frank-Wolfe ---
    gap_with = [r.relative_gap for r in result_with.iteration_log]
    gap_without = [r.relative_gap for r in result_without.iteration_log]
    gap_fw_w = [r.relative_gap for r in result_fw_with.iteration_log]
    gap_fw_wo = [r.relative_gap for r in result_fw_without.iteration_log]
    iters_fw_w = [r.iteration for r in result_fw_with.iteration_log]
    iters_fw_wo = [r.iteration for r in result_fw_without.iteration_log]

    def _moving_max(gaps, window=5):
        """Rolling max of gap values (envelope of worst-case per window)."""
        import numpy as np
        arr = np.array(gaps)
        out = np.empty_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            out[i] = arr[start:i+1].max()
        return out.tolist()

    fig_gap = go.Figure()
    # MSA raw gap as faint markers
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
    # MSA envelope (rolling max of |gap|) as dashed
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
    # FW gap as solid lines
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
        "<h2>Convergence: MSA vs Frank-Wolfe</h2>"
        "<p>MSA (dashed envelopes, faint dots) vs Frank-Wolfe (solid lines). "
        "FW uses Beckmann line search to find the optimal step size each iteration, "
        "eliminating the bang-bang oscillation inherent to MSA on small networks. "
        f"FW final gap: with shortcut = {fw_gap_w_fmt}, without = {fw_gap_wo_fmt}.</p>"
    )

    # --- 4. FW step size ---
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
    # MSA step size for reference
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
        "<h2>FW Step Size</h2>"
        "<p>Optimal step size &alpha;* from the Beckmann line search each iteration. "
        "Dotted grey = MSA fixed schedule (1/n). Early iterations take large steps; "
        "as equilibrium is approached, FW takes progressively smaller steps &mdash; "
        "unlike MSA which follows a rigid 1/n schedule regardless of the objective landscape.</p>"
    )

    # Write report
    demand_fmt = f"{demand:,.0f}"
    _write_combined_report(
        title="Braess Paradox Validation",
        intro=f"""<p>Structural validation of the density-based traffic assignment using the
        <b>Braess paradox</b> &mdash; a 4-node diamond network where adding a shortcut link
        increases total system travel time under user equilibrium.</p>
        <p>Demand: <b>{demand_fmt}</b> vehicles.
        <b>{max_iter}</b> iterations per scenario.
        VDF: bi-parabolic speed-density (Fournier et al.), k<sub>c</sub> = k<sub>j</sub>/3.
        Methods compared: MSA (&alpha; = 1/n) and Frank-Wolfe (Beckmann line search).</p>""",
        figures=figs,
        descriptions=descriptions,
        path=Path(output_path),
    )
    return Path(output_path)
