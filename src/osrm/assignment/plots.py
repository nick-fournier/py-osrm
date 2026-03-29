"""Validation and diagnostic plots for traffic assignment.

Generates interactive Plotly HTML plots for:
1. VDF theory validation (pure math)
2. Assignment convergence diagnostics (per-iteration)
3. Network state diagnostics (post-assignment)
4. Benchmark validation (Braess, Sioux Falls)

Usage:
    from osrm.assignment import plots
    plots.vdf_theory()                  # opens browser / saves HTML
    plots.convergence(iteration_log)    # after assignment run
    plots.network_diagnostics(state)    # post-assignment state
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from osrm.assignment.vdf import BiParabolicVDF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig: go.Figure, path: Optional[str] = None) -> go.Figure:
    """Save to HTML file or show in browser."""
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(p), include_plotlyjs="cdn")
    else:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 1. VDF Theory Plots
# ---------------------------------------------------------------------------

def vdf_speed_density(
    v_f: float = 60.0,
    k_j: float = 150.0,
    vdf: BiParabolicVDF | None = None,
    path: Optional[str] = None,
) -> go.Figure:
    """Plot speed-density curve with both branches and k_c annotation.

    Parameters
    ----------
    v_f : float
        Free-flow speed (km/h).
    k_j : float
        Jam density (veh/km).
    vdf : BiParabolicVDF, optional
        VDF instance. Uses default if None.
    path : str, optional
        Save to HTML file path. Shows in browser if None.
    """
    vdf = vdf or BiParabolicVDF()
    k_c = vdf.kc_ratio * k_j
    v_c = v_f / 2.0

    k = np.linspace(0, k_j, 500)
    v_f_arr = np.full_like(k, v_f)
    k_j_arr = np.full_like(k, k_j)
    v = vdf.density_to_speed(k, v_f_arr, k_j_arr)

    # Split into branches for coloring
    mask_unc = k <= k_c
    mask_con = k > k_c

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k[mask_unc], y=v[mask_unc],
        mode="lines", name="Uncongested",
        line=dict(color="#2196F3", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=k[mask_con], y=v[mask_con],
        mode="lines", name="Congested",
        line=dict(color="#F44336", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[k_c], y=[v_c],
        mode="markers+text", name=f"k_c = {k_c:.0f}",
        marker=dict(size=12, color="#FF9800", symbol="diamond"),
        text=[f"k_c={k_c:.0f}, v_c={v_c:.0f}"],
        textposition="top right",
    ))
    fig.add_hline(y=vdf.min_speed_kmh, line_dash="dot",
                  annotation_text=f"Floor = {vdf.min_speed_kmh} km/h",
                  line_color="gray")

    fig.update_layout(
        title=f"Bi-Parabolic Speed–Density (v_f={v_f}, k_j={k_j})",
        xaxis_title="Density k (veh/km)",
        yaxis_title="Speed v (km/h)",
        template="plotly_white",
        legend=dict(x=0.7, y=0.95),
    )
    return _save_or_show(fig, path)


def vdf_flow_density(
    v_f: float = 60.0,
    k_j: float = 150.0,
    vdf: BiParabolicVDF | None = None,
    path: Optional[str] = None,
) -> go.Figure:
    """Plot flow-density fundamental diagram (MFD parabolas)."""
    vdf = vdf or BiParabolicVDF()
    k_c = vdf.kc_ratio * k_j
    q_c = v_f * k_c / 2.0

    k = np.linspace(0, k_j, 500)
    v_f_arr = np.full_like(k, v_f)
    k_j_arr = np.full_like(k, k_j)
    q = vdf.density_to_flow(k, v_f_arr, k_j_arr)

    mask_unc = k <= k_c
    mask_con = k > k_c

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k[mask_unc], y=q[mask_unc],
        mode="lines", name="Uncongested",
        line=dict(color="#2196F3", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=k[mask_con], y=q[mask_con],
        mode="lines", name="Congested",
        line=dict(color="#F44336", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[k_c], y=[q_c],
        mode="markers+text", name=f"Capacity q_c = {q_c:.0f}",
        marker=dict(size=12, color="#FF9800", symbol="diamond"),
        text=[f"q_c={q_c:.0f} veh/hr"],
        textposition="top right",
    ))

    fig.update_layout(
        title=f"Bi-Parabolic Flow–Density MFD (v_f={v_f}, k_j={k_j})",
        xaxis_title="Density k (veh/km)",
        yaxis_title="Flow q (veh/hr)",
        template="plotly_white",
        legend=dict(x=0.6, y=0.95),
    )
    return _save_or_show(fig, path)


def vdf_inverse_accuracy(
    v_f: float = 60.0,
    k_j: float = 150.0,
    n_points: int = 200,
    vdf: BiParabolicVDF | None = None,
    path: Optional[str] = None,
) -> go.Figure:
    """Plot round-trip inverse accuracy: q → k(q) → q(k) scatter."""
    vdf = vdf or BiParabolicVDF()
    q_c = vdf.capacity_flow(np.array([v_f]), np.array([k_j]))[0]

    q_in = np.linspace(0, q_c * 0.999, n_points)
    v_f_arr = np.full_like(q_in, v_f)
    k_j_arr = np.full_like(q_in, k_j)
    k = vdf.flow_to_density(q_in, v_f_arr, k_j_arr)
    q_out = vdf.density_to_flow(k, v_f_arr, k_j_arr)

    error = np.abs(q_out - q_in)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["q_in vs q_out", "Absolute Error"])

    fig.add_trace(go.Scatter(
        x=q_in, y=q_out, mode="markers",
        marker=dict(size=3, color="#2196F3"),
        name="Round-trip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, q_c], y=[0, q_c], mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=q_in, y=error, mode="markers",
        marker=dict(size=3, color="#F44336"),
        name="Error",
    ), row=1, col=2)

    fig.update_xaxes(title_text="q_in (veh/hr)", row=1, col=1)
    fig.update_yaxes(title_text="q_out (veh/hr)", row=1, col=1)
    fig.update_xaxes(title_text="q_in (veh/hr)", row=1, col=2)
    fig.update_yaxes(title_text="|q_out - q_in|", exponentformat="e", row=1, col=2)

    fig.update_layout(
        title=f"VDF Inverse Round-Trip Accuracy (max err: {error.max():.2e})",
        template="plotly_white",
        showlegend=False,
    )
    return _save_or_show(fig, path)


def vdf_multi_class(
    classes: dict[str, tuple[float, float]] | None = None,
    vdf: BiParabolicVDF | None = None,
    path: Optional[str] = None,
) -> go.Figure:
    """Overlay speed-density curves for multiple road classes.

    Parameters
    ----------
    classes : dict
        Road class name → (v_f, k_j). Defaults to typical values.
    """
    vdf = vdf or BiParabolicVDF()
    if classes is None:
        classes = {
            "Motorway (v_f=110, k_j=450)": (110.0, 450.0),
            "Primary (v_f=60, k_j=260)": (60.0, 260.0),
            "Residential (v_f=30, k_j=100)": (30.0, 100.0),
        }

    fig = go.Figure()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    for i, (name, (v_f, k_j)) in enumerate(classes.items()):
        k = np.linspace(0, k_j, 300)
        v = vdf.density_to_speed(k, np.full_like(k, v_f), np.full_like(k, k_j))
        fig.add_trace(go.Scatter(
            x=k, y=v, mode="lines", name=name,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title="Speed–Density by Road Class",
        xaxis_title="Density k (veh/km)",
        yaxis_title="Speed v (km/h)",
        template="plotly_white",
    )
    return _save_or_show(fig, path)


def vdf_theory(output_dir: str = "docs/plots") -> Path:
    """Generate a single combined VDF theory validation report.

    Returns path to the saved HTML file.
    """
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "vdf_theory_report.html"

    vdf = BiParabolicVDF()
    v_f, k_j = 60.0, 150.0
    k_c = vdf.kc_ratio * k_j

    figs = [
        vdf_speed_density(v_f=v_f, k_j=k_j, vdf=vdf),
        vdf_flow_density(v_f=v_f, k_j=k_j, vdf=vdf),
        vdf_inverse_accuracy(v_f=v_f, k_j=k_j, vdf=vdf),
        vdf_multi_class(vdf=vdf),
    ]

    descriptions = [
        f"""<h2>1. Speed–Density Relationship</h2>
        <p>The bi-parabolic model (Fournier) defines speed as a function of density using
        two branches joined at critical density k<sub>c</sub> = k<sub>j</sub>/3 = {k_c:.0f} veh/km.
        The <b>uncongested branch</b> (blue) is linear in v-k space:
        v(k) = q<sub>c</sub>(2k<sub>c</sub> − k) / k<sub>c</sub>². The <b>congested branch</b> (red) is
        parabolic in q-k space, yielding a nonlinear speed drop. At k = 0, v = v<sub>f</sub> = {v_f:.0f} km/h.
        At k = k<sub>c</sub>, v = v<sub>f</sub>/2 = {v_f/2:.0f} km/h. The junction is C¹-continuous
        (matching value and slope). A floor of {vdf.min_speed_kmh} km/h prevents zero speeds.</p>""",

        f"""<h2>2. Flow–Density Fundamental Diagram (MFD)</h2>
        <p>This IS the macroscopic fundamental diagram. Both branches are downward-opening
        parabolas in q-k space with vertex at (k<sub>c</sub>, q<sub>c</sub>). Capacity flow
        q<sub>c</sub> = v<sub>f</sub> · k<sub>c</sub> / 2 = {v_f * k_c / 2:.0f} veh/hr occurs at critical
        density. The uncongested branch (left of k<sub>c</sub>) is the operating regime for
        equilibrium assignment — the congested branch represents breakdown conditions
        where adding vehicles reduces throughput.</p>""",

        """<h2>3. Inverse Round-Trip Accuracy</h2>
        <p>A key advantage of this VDF over BPR: the flow-to-density inversion has a
        <b>closed-form solution</b> via the quadratic formula — no Newton solver needed.
        Left panel: q<sub>in</sub> vs q<sub>out</sub> after q → k(q) → q(k) round-trip
        (should fall exactly on the diagonal). Right panel: absolute error, which should
        be near machine epsilon (~10<sup>-10</sup>). This confirms the vectorized NumPy
        implementation is numerically exact.</p>""",

        """<h2>4. Speed–Density by Road Class</h2>
        <p>The bi-parabolic model is "parameter-light" — only v<sub>f</sub> (free-flow speed)
        and k<sub>j</sub> (jam density) are needed per link. k<sub>c</sub> = k<sub>j</sub>/3 is derived,
        not calibrated. This overlay shows how different road classes produce different
        curves from just those two inputs. Motorways have higher v<sub>f</sub> and k<sub>j</sub>
        (more lanes × higher jam density per lane), while residential streets have lower
        values of both. The shape is consistent across classes — only the scale changes.</p>""",
    ]

    _write_combined_report(
        title="Bi-Parabolic VDF Theory Validation",
        intro="""<p>These plots validate the bi-parabolic flow-density Volume Delay Function
        (Fournier) implemented in <code>osrm.assignment.vdf.BiParabolicVDF</code>.
        The model uses two parabolic branches in q-k space, requiring only free-flow speed
        (v<sub>f</sub>) and jam density (k<sub>j</sub>) as inputs. All plots use default parameters:
        v<sub>f</sub> = 60 km/h, k<sub>j</sub> = 150 veh/km, k<sub>c</sub> = k<sub>j</sub>/3 = 50 veh/km.</p>""",
        figures=figs,
        descriptions=descriptions,
        path=path,
    )
    return path


def _write_combined_report(
    title: str,
    intro: str,
    figures: list[go.Figure],
    descriptions: list[str],
    path: Path,
) -> None:
    """Write multiple Plotly figures into a single HTML report with descriptions."""
    fig_htmls = []
    for i, fig in enumerate(figures):
        fig.update_layout(margin=dict(l=60, r=40, t=50, b=50))
        fig_htmls.append(
            fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig-{i}")
        )

    sections = []
    for desc, fig_html in zip(descriptions, fig_htmls):
        sections.append(f"""
        <section style="margin-bottom: 40px;">
            {desc}
            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; margin-top: 12px;">
                {fig_html}
            </div>
        </section>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1100px;
            margin: 0 auto;
            padding: 20px 40px;
            color: #333;
            line-height: 1.6;
        }}
        h1 {{
            border-bottom: 2px solid #2196F3;
            padding-bottom: 10px;
            color: #1565C0;
        }}
        h2 {{
            color: #1976D2;
            margin-top: 0;
        }}
        p {{
            font-size: 15px;
            max-width: 900px;
        }}
        code {{
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 14px;
        }}
        section {{
            border-left: 3px solid #e3f2fd;
            padding-left: 20px;
        }}
        .meta {{
            color: #777;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="meta">Generated by py-osrm traffic assignment module</div>
    {intro}
    {"".join(sections)}
</body>
</html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def convergence_report(
    iteration_log: dict,
    network_state=None,
    vdf: BiParabolicVDF | None = None,
    path: str = "docs/plots/assignment_report.html",
) -> Path:
    """Generate a combined convergence + network diagnostics report.

    Parameters
    ----------
    iteration_log : dict
        Keys: "iteration", "relative_gap", "tstt", optionally "max_flow_delta".
    network_state : NetworkState, optional
        If provided, adds network diagnostic plots.
    vdf : BiParabolicVDF, optional
    path : str
        Output HTML path.
    """
    vdf = vdf or BiParabolicVDF()
    figs = []
    descriptions = []

    # Convergence plot
    figs.append(convergence(iteration_log))
    descriptions.append("""<h2>1. Convergence Diagnostics</h2>
    <p>Top: <b>Relative gap</b> — the standard convergence measure for equilibrium
    assignment. Should decrease monotonically toward zero. The dotted green line at
    0.01 is the conventional "converged" threshold. Middle: <b>Total system travel
    time (TSTT)</b> — should stabilize as equilibrium is reached. If TSTT oscillates,
    the step size (MSA weight) may be too aggressive.</p>""")

    if network_state is not None:
        q_c = vdf.capacity_flow(network_state.freeflow_kmh, network_state.jam_density)

        figs.append(flow_vs_capacity(network_state.flow_vph, q_c))
        descriptions.append("""<h2>2. Flow vs Capacity</h2>
        <p>Each point is one directed link. Points above the diagonal (V/C > 1.0) are
        <b>oversaturated</b> — assigned flow exceeds theoretical capacity. Blue = uncongested
        (V/C < 0.8), orange = near-capacity (0.8–1.0), red = oversaturated. A healthy
        assignment should have most points below the diagonal, with a few near-capacity
        links on major corridors.</p>""")

        figs.append(speed_reduction(network_state.speed_kmh, network_state.freeflow_kmh))
        descriptions.append("""<h2>3. Speed Reduction Distribution</h2>
        <p>Histogram of v/v<sub>f</sub> ratio across all links. At v/v<sub>f</sub> = 1.0,
        links are at free-flow speed (no congestion). The dashed line at 0.5 marks
        v<sub>c</sub> — the critical speed at capacity. A well-loaded network should show
        a peak near 1.0 (most links uncongested) with a tail toward lower ratios on
        congested corridors.</p>""")

        figs.append(observed_vs_mfd(
            network_state.flow_vph, network_state.density_vpkm,
            vdf=vdf,
        ))
        descriptions.append("""<h2>4. Assigned Points vs Theoretical MFD</h2>
        <p>Red scatter points are (density, flow) pairs from the assignment overlaid
        on the theoretical bi-parabolic fundamental diagram (blue curve). In a consistent
        assignment, all points should fall on or near the uncongested branch of the MFD.
        Points above the curve indicate inconsistency between the VDF and the assigned state
        (possible numerical issue). Points to the right of k<sub>c</sub> indicate links
        operating in the congested regime.</p>""")

    _write_combined_report(
        title="Traffic Assignment Report",
        intro="<p>Post-assignment diagnostics from py-osrm traffic assignment.</p>",
        figures=figs,
        descriptions=descriptions,
        path=Path(path),
    )
    return Path(path)


# ---------------------------------------------------------------------------
# 2. Assignment Convergence Diagnostics
# ---------------------------------------------------------------------------

def convergence(
    iteration_log: dict,
    path: Optional[str] = None,
) -> go.Figure:
    """Plot assignment convergence diagnostics.

    Parameters
    ----------
    iteration_log : dict
        Must contain keys:
        - "iteration": list[int]
        - "relative_gap": list[float]
        - "tstt": list[float]  (total system travel time)
        Optionally:
        - "max_flow_delta": list[float]
    """
    iters = iteration_log["iteration"]
    gap = iteration_log["relative_gap"]
    tstt = iteration_log["tstt"]

    n_rows = 3 if "max_flow_delta" in iteration_log else 2
    titles = ["Relative Gap", "Total System Travel Time"]
    if n_rows == 3:
        titles.append("Max Link Flow Delta")

    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=titles,
                        vertical_spacing=0.08)

    fig.add_trace(go.Scatter(
        x=iters, y=gap, mode="lines+markers",
        line=dict(color="#F44336", width=2),
        marker=dict(size=6), name="Rel. Gap",
    ), row=1, col=1)
    fig.add_hline(y=0.01, line_dash="dot", line_color="green",
                  annotation_text="Target (0.01)", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=iters, y=tstt, mode="lines+markers",
        line=dict(color="#2196F3", width=2),
        marker=dict(size=6), name="TSTT",
    ), row=2, col=1)

    if n_rows == 3:
        fig.add_trace(go.Scatter(
            x=iters, y=iteration_log["max_flow_delta"],
            mode="lines+markers",
            line=dict(color="#FF9800", width=2),
            marker=dict(size=6), name="Max Δq",
        ), row=3, col=1)

    fig.update_layout(
        title="Assignment Convergence",
        template="plotly_white",
        height=250 * n_rows,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Iteration", row=n_rows, col=1)
    return _save_or_show(fig, path)


def flow_delta_histogram(
    flow_deltas: np.ndarray,
    iteration: int,
    path: Optional[str] = None,
) -> go.Figure:
    """Histogram of per-link flow changes between iterations."""
    fig = go.Figure(go.Histogram(
        x=flow_deltas, nbinsx=100,
        marker_color="#2196F3",
    ))
    fig.update_layout(
        title=f"Link Flow Delta Distribution (Iteration {iteration})",
        xaxis_title="Δq (veh/hr)",
        yaxis_title="Count",
        template="plotly_white",
    )
    return _save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 3. Network State Diagnostics
# ---------------------------------------------------------------------------

def flow_vs_capacity(
    flow: np.ndarray,
    capacity: np.ndarray,
    path: Optional[str] = None,
) -> go.Figure:
    """Scatter plot of link flow vs capacity. Points above diagonal are oversaturated."""
    vc_ratio = flow / np.where(capacity > 0, capacity, 1.0)
    colors = np.where(vc_ratio > 1.0, "#F44336", np.where(vc_ratio > 0.8, "#FF9800", "#2196F3"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=capacity, y=flow, mode="markers",
        marker=dict(size=4, color=colors, opacity=0.6),
        text=[f"V/C={r:.2f}" for r in vc_ratio],
        hovertemplate="Capacity: %{x:.0f}<br>Flow: %{y:.0f}<br>%{text}",
        showlegend=False,
    ))
    max_val = max(flow.max(), capacity.max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(dash="dash", color="gray"),
        name="V/C = 1.0",
    ))

    fig.update_layout(
        title="Link Flow vs Capacity",
        xaxis_title="Capacity q_c (veh/hr)",
        yaxis_title="Assigned Flow q (veh/hr)",
        template="plotly_white",
    )
    return _save_or_show(fig, path)


def speed_reduction(
    speed: np.ndarray,
    freeflow: np.ndarray,
    path: Optional[str] = None,
) -> go.Figure:
    """Histogram of v/v_f ratio across all links."""
    ratio = speed / np.where(freeflow > 0, freeflow, 1.0)

    fig = go.Figure(go.Histogram(
        x=ratio, nbinsx=50,
        marker_color="#2196F3",
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color="green",
                  annotation_text="Free-flow")
    fig.add_vline(x=0.5, line_dash="dot", line_color="orange",
                  annotation_text="v_c (50%)")

    fig.update_layout(
        title="Speed Reduction Distribution (v / v_f)",
        xaxis_title="v / v_f",
        yaxis_title="Number of Links",
        template="plotly_white",
    )
    return _save_or_show(fig, path)


def observed_vs_mfd(
    flow: np.ndarray,
    density: np.ndarray,
    v_f: float = 60.0,
    k_j: float = 150.0,
    vdf: BiParabolicVDF | None = None,
    path: Optional[str] = None,
) -> go.Figure:
    """Overlay observed (flow, density) points on theoretical MFD curve."""
    vdf = vdf or BiParabolicVDF()

    k_theory = np.linspace(0, k_j, 300)
    q_theory = vdf.density_to_flow(
        k_theory, np.full_like(k_theory, v_f), np.full_like(k_theory, k_j)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_theory, y=q_theory, mode="lines",
        line=dict(color="#2196F3", width=2),
        name="Theoretical MFD",
    ))
    fig.add_trace(go.Scatter(
        x=density, y=flow, mode="markers",
        marker=dict(size=4, color="#F44336", opacity=0.5),
        name="Observed (assignment)",
    ))

    fig.update_layout(
        title="Assigned Points vs Theoretical MFD",
        xaxis_title="Density k (veh/km)",
        yaxis_title="Flow q (veh/hr)",
        template="plotly_white",
    )
    return _save_or_show(fig, path)


def network_diagnostics(
    network_state,
    vdf: BiParabolicVDF | None = None,
    output_dir: str = "docs/plots",
) -> list[Path]:
    """Generate all network state diagnostic plots.

    Parameters
    ----------
    network_state : NetworkState
        Post-assignment network state.
    """
    vdf = vdf or BiParabolicVDF()
    d = Path(output_dir)

    q_c = vdf.capacity_flow(network_state.freeflow_kmh, network_state.jam_density)

    flow_vs_capacity(
        network_state.flow_vph, q_c,
        path=str(d / "flow_vs_capacity.html"),
    )
    speed_reduction(
        network_state.speed_kmh, network_state.freeflow_kmh,
        path=str(d / "speed_reduction.html"),
    )
    observed_vs_mfd(
        network_state.flow_vph, network_state.density_vpkm,
        path=str(d / "observed_vs_mfd.html"),
    )

    return [
        d / "flow_vs_capacity.html",
        d / "speed_reduction.html",
        d / "observed_vs_mfd.html",
    ]


# ---------------------------------------------------------------------------
# 4. Benchmark Validation Plots
# ---------------------------------------------------------------------------

def path_cost_comparison(
    od_pairs: list[str],
    path_costs: dict[str, list[float]],
    path: Optional[str] = None,
) -> go.Figure:
    """Box plot of path costs per OD pair at equilibrium.

    At Wardrop equilibrium, all used paths for an OD pair should have equal cost.

    Parameters
    ----------
    od_pairs : list[str]
        OD pair labels.
    path_costs : dict
        OD label → list of path travel times for used paths.
    """
    fig = go.Figure()
    for od in od_pairs:
        costs = path_costs[od]
        fig.add_trace(go.Box(
            y=costs, name=od,
            boxmean=True,
        ))

    fig.update_layout(
        title="Path Cost Comparison at Equilibrium (Wardrop Check)",
        yaxis_title="Path Travel Time (s)",
        template="plotly_white",
    )
    return _save_or_show(fig, path)


def assigned_vs_published(
    assigned: np.ndarray,
    published: np.ndarray,
    labels: Sequence[str] | None = None,
    path: Optional[str] = None,
) -> go.Figure:
    """Scatter: assigned link flows vs published equilibrium flows.

    For Sioux Falls validation. Reports correlation coefficient.
    """
    r = np.corrcoef(assigned, published)[0, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=published, y=assigned, mode="markers",
        marker=dict(size=6, color="#2196F3", opacity=0.7),
        text=labels,
        hovertemplate="Published: %{x:.0f}<br>Assigned: %{y:.0f}",
        showlegend=False,
    ))
    max_val = max(published.max(), assigned.max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(dash="dash", color="gray"),
        name="y = x",
    ))

    fig.update_layout(
        title=f"Assigned vs Published Link Flows (r = {r:.3f})",
        xaxis_title="Published Flow (veh/hr)",
        yaxis_title="Assigned Flow (veh/hr)",
        template="plotly_white",
    )
    return _save_or_show(fig, path)


def geh_distribution(
    model_flow: np.ndarray,
    observed_flow: np.ndarray,
    path: Optional[str] = None,
) -> go.Figure:
    """Histogram of GEH statistic across all links.

    GEH = sqrt(2(M-C)² / (M+C))
    GEH < 5 for >85% of links is industry standard for validation.
    """
    denom = np.where((model_flow + observed_flow) > 0,
                     model_flow + observed_flow, 1.0)
    geh = np.sqrt(2.0 * (model_flow - observed_flow) ** 2 / denom)
    pct_good = np.mean(geh < 5.0) * 100.0

    fig = go.Figure(go.Histogram(
        x=geh, nbinsx=50,
        marker_color="#2196F3",
    ))
    fig.add_vline(x=5.0, line_dash="dash", line_color="red",
                  annotation_text=f"GEH=5 ({pct_good:.1f}% below)")

    fig.update_layout(
        title=f"GEH Statistic Distribution ({pct_good:.1f}% < 5)",
        xaxis_title="GEH",
        yaxis_title="Number of Links",
        template="plotly_white",
    )
    return _save_or_show(fig, path)
