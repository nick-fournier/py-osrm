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

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)
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


def _vdf_near_jam_detail(
    v_f: float = 60.0,
    k_j: float = 150.0,
    vdf: BiParabolicVDF | None = None,
) -> go.Figure:
    """Plot speed and travel-time sensitivity near jam density.

    Shows the steep gradient in the congested regime that causes
    convergence difficulties for Frank-Wolfe assignment.
    """
    vdf = vdf or BiParabolicVDF()

    # Zoom: k/k_j from 0.5 to 1.0
    frac = np.linspace(0.5, 1.0, 500)
    k = frac * k_j
    v_f_arr = np.full_like(k, v_f)
    k_j_arr = np.full_like(k, k_j)
    v = vdf.density_to_speed(k, v_f_arr, k_j_arr)

    # Travel time per km (seconds)
    tt_per_km = 3600.0 / np.maximum(v, 0.001)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Speed near k_j", "Travel time per km near k_j"],
    )

    fig.add_trace(go.Scatter(
        x=frac, y=v, mode="lines",
        line=dict(color="#F44336", width=2),
        name="Speed",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=frac, y=tt_per_km, mode="lines",
        line=dict(color="#FF9800", width=2),
        name="Travel time/km",
    ), row=1, col=2)

    # Annotate key thresholds
    for threshold, label in [(0.9, "90%"), (0.95, "95%"), (0.99, "99%")]:
        kk = np.array([threshold * k_j])
        vv = vdf.density_to_speed(kk, np.array([v_f]), np.array([k_j]))[0]
        tt = 3600.0 / max(vv, 0.001)
        fig.add_trace(go.Scatter(
            x=[threshold], y=[vv], mode="markers+text",
            marker=dict(size=8, color="#333"),
            text=[f"{label}: {vv:.1f} km/h"],
            textposition="top left",
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[threshold], y=[tt], mode="markers+text",
            marker=dict(size=8, color="#333"),
            text=[f"{label}: {tt:.0f}s/km"],
            textposition="top left",
            showlegend=False,
        ), row=1, col=2)

    fig.update_xaxes(title_text="k / k_j", row=1, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_xaxes(title_text="k / k_j", row=1, col=2)
    fig.update_yaxes(title_text="Travel time (s/km)", type="log", row=1, col=2)
    fig.update_layout(
        title="Near-Jam Behaviour: Speed Collapse and Travel-Time Explosion",
        template="plotly_white",
        showlegend=False,
        height=400,
    )
    return fig


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

    # Match axes on the round-trip scatter — same ticks on both axes
    max_q = float(q_c * 1.05)
    tick_step = 250.0
    fig.update_xaxes(range=[0, max_q], dtick=tick_step, row=1, col=1)
    fig.update_yaxes(range=[0, max_q], dtick=tick_step, scaleanchor="x", scaleratio=1, row=1, col=1)

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

    Uses the same parameters as the assignment loop defaults so the
    report reflects actual operational behaviour.

    Returns path to the saved HTML file.
    """
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "vdf_theory_report.html"

    # Use assignment-actual min_speed (0.01 km/h, not VDF default 5.0)
    from osrm.assignment.assignment_loop import AssignmentConfig
    cfg = AssignmentConfig()
    vdf = BiParabolicVDF(
        kc_ratio=cfg.vdf_kc_ratio,
        min_speed_kmh=cfg.vdf_min_speed_kmh,
    )
    v_f, k_j = 60.0, cfg.default_jam_density_per_lane * cfg.default_n_lanes
    k_c = vdf.kc_ratio * k_j
    q_c = v_f * k_c / 2.0

    figs = [
        vdf_speed_density(v_f=v_f, k_j=k_j, vdf=vdf),
        vdf_flow_density(v_f=v_f, k_j=k_j, vdf=vdf),
        _vdf_near_jam_detail(v_f=v_f, k_j=k_j, vdf=vdf),
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
        (matching value and slope). A floor of {vdf.min_speed_kmh} km/h prevents division by zero.</p>""",

        f"""<h2>2. Flow–Density Fundamental Diagram (MFD)</h2>
        <p>This IS the macroscopic fundamental diagram. Both branches are downward-opening
        parabolas in q-k space with vertex at ($k_c$, $q_c$). Capacity flow
        $q_c = v_f \\cdot k_c / 2$ = {q_c:.0f} veh/hr occurs at critical
        density. The uncongested branch (left of $k_c$) is the operating regime for
        equilibrium assignment — the congested branch represents breakdown conditions
        where adding vehicles reduces throughput.</p>""",

        f"""<h2>3. Near-Jam Behaviour (k → k<sub>j</sub>)</h2>
        <p>This panel zooms into the congested branch near jam density to show the steep
        speed gradient that impacts convergence. At k/k<sub>j</sub> = 0.90, speed is just
        {vdf.density_to_speed(np.array([0.9*k_j]), np.array([v_f]), np.array([k_j]))[0]:.1f} km/h.
        At k/k<sub>j</sub> = 0.99 it drops to
        {vdf.density_to_speed(np.array([0.99*k_j]), np.array([v_f]), np.array([k_j]))[0]:.2f} km/h.
        This extreme sensitivity means a small density change near k<sub>j</sub> produces a
        large travel-time change — creating a near-discontinuity that Frank-Wolfe's linear
        search direction struggles to navigate.</p>
        <p>Density is <b>uncapped</b> above k<sub>j</sub>: the VDF's speed floor
        ({vdf.min_speed_kmh} km/h) handles oversaturated links. Links exceeding k<sub>j</sub>
        represent virtual queue / spillback — physically impossible density but
        mathematically stable.</p>""",

        """<h2>4. Inverse Round-Trip Accuracy</h2>
        <p>A key advantage of this VDF over BPR: the flow-to-density inversion has a
        <b>closed-form solution</b> via the quadratic formula — no Newton solver needed.
        Left panel: q<sub>in</sub> vs q<sub>out</sub> after q → k(q) → q(k) round-trip
        (should fall exactly on the diagonal). Right panel: absolute error, which should
        be near machine epsilon (~10<sup>-10</sup>). This confirms the vectorized NumPy
        implementation is numerically exact.</p>""",

        """<h2>5. Speed–Density by Road Class</h2>
        <p>The bi-parabolic model is "parameter-light" — only v<sub>f</sub> (free-flow speed)
        and k<sub>j</sub> (jam density) are needed per link. k<sub>c</sub> = k<sub>j</sub>/3 is derived,
        not calibrated. This overlay shows how different road classes produce different
        curves from just those two inputs. Motorways have higher v<sub>f</sub> and k<sub>j</sub>
        (more lanes × higher jam density per lane), while residential streets have lower
        values of both. The shape is consistent across classes — only the scale changes.</p>""",
    ]

    _write_combined_report(
        title="Bi-Parabolic VDF Theory Validation",
        intro=f"""
        <div class="equations">
        <h3>Model Equations</h3>
        <p><b>Derived parameters:</b>
        $k_c = k_j / 3$ (critical density) &nbsp;|&nbsp;
        $q_c = v_f \\cdot k_c / 2$ (capacity flow) &nbsp;|&nbsp;
        $v_c = v_f / 2$ (critical speed)</p>

        <p><b>Uncongested branch</b> ($k \\leq k_c$): &emsp;
        $v(k) = \\dfrac{{q_c \\left(2 k_c - k\\right)}}{{k_c^2}}$</p>

        <p><b>Congested branch</b> ($k > k_c$): &emsp;
        $v(k) = \\dfrac{{q_c}}{{k}} \\left[1 - \\dfrac{{(k - k_c)^2}}{{(k_j - k_c)^2}}\\right]$</p>

        <p><b>Fundamental identity:</b> &emsp; $q = k \\cdot v(k)$</p>

        <p><b>Wardrop relative gap:</b> &emsp;
        $\\text{{gap}} = \\dfrac{{\\sum_a V_a \\cdot t_a}}{{\\sum_{{rs}} d_{{rs}} \\cdot \\pi_{{rs}}}} - 1$
        &emsp; where $t_a = L_a / v_a$ is link travel time, $\\pi_{{rs}}$ is shortest-path cost.</p>
        </div>

        <p><b>Parameters (matching assignment defaults):</b>
        $v_f$ = {v_f:.0f} km/h,
        $k_j$ = {k_j:.0f} veh/km,
        $k_c$ = {k_c:.0f} veh/km,
        $q_c$ = {q_c:.0f} veh/hr,
        min speed = {vdf.min_speed_kmh} km/h.</p>""",
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
        if fig is not None:
            fig.update_layout(
                margin=dict(l=60, r=40, t=50, b=50),
                xaxis=dict(fixedrange=True),
                yaxis=dict(fixedrange=True),
                dragmode=False,
            )
            # Lock axes on subplots too
            for key in list(fig.layout.to_plotly_json().keys()):
                if key.startswith("xaxis") or key.startswith("yaxis"):
                    fig.layout[key]["fixedrange"] = True
            fig_htmls.append(
                fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig-{i}")
            )
        else:
            fig_htmls.append("")

    sections = []
    for desc, fig_html in zip(descriptions, fig_htmls):
        inner = ""
        if fig_html:
            inner = f"""
            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; margin-top: 12px;">
                {fig_html}
            </div>"""
        sections.append(f"""
        <section style="margin-bottom: 40px;">
            {desc}{inner}
        </section>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
    <script>
        MathJax = {{ tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }} }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
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
        .equations {{
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px 30px;
            margin: 20px 0;
        }}
        .equations h3 {{
            color: #1976D2;
            margin-top: 0;
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
        Keys: "iteration", "relative_gap", "tstt", optionally "max_density_delta".
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
        - "max_density_delta": list[float]
    """
    iters = iteration_log["iteration"]
    gap = iteration_log["relative_gap"]
    tstt = iteration_log["tstt"]

    n_rows = 3 if "max_density_delta" in iteration_log else 2
    titles = ["Relative Gap", "Total System Travel Time"]
    if n_rows == 3:
        titles.append("Max Link Density Delta")

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
            x=iters, y=iteration_log["max_density_delta"],
            mode="lines+markers",
            line=dict(color="#FF9800", width=2),
            marker=dict(size=6), name="Max Δk",
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


# ---------------------------------------------------------------------------
# Generic TNTP validation report
# ---------------------------------------------------------------------------

def generate_validation_report(
    network_name: str,
    prepare_fn,
    run_fn,
    copy_fn,
    tmp_path: Path,
    *,
    output_path: str = "docs/plots/validation.html",
    max_iter: int = 50,
    detail_scale: float = 0.15,
    sweep_scales: Sequence[float] | None = None,
    vc_scales: Sequence[float] | None = None,
    intro_html: str = "",
) -> Path:
    """Generate a standard validation report for any TNTP network.

    Produces an interactive HTML report with:

    0. Network topology map
    1. Demand scaling sweep (oversaturation, speed, TSTT)
    2. FW vs MSA convergence comparison
    3. Per-link state table
    4. Flow and travel time correlation vs BPR reference
    5. V/C scatter at multiple demand levels

    Parameters
    ----------
    network_name : str
        Human-readable name (e.g. "Sioux Falls", "Anaheim").
    prepare_fn : callable
        ``(tmp_path: Path) -> (base_path: str, meta: dict)``
    run_fn : callable
        ``(base: str, meta: dict, max_iter: int, method: str,
        demand_scale: float) -> AssignmentResult``
    copy_fn : callable
        ``(base_path: str, run_dir: Path) -> base_path: str``
    tmp_path : Path
        Working directory for temporary OSRM files.
    output_path : str
        Where to write the HTML report.
    max_iter : int
        Assignment iterations per run.
    detail_scale : float
        Primary demand fraction for detailed analysis.
    sweep_scales : sequence of float, optional
        Demand fractions for the scaling sweep.
    vc_scales : sequence of float, optional
        Demand fractions for V/C scatter plot.
    intro_html : str
        Additional HTML to insert after the auto-generated intro.

    Returns
    -------
    Path to the generated report.
    """
    if sweep_scales is None:
        sweep_scales = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
    if vc_scales is None:
        vc_scales = [0.05, 0.10, 0.15, 0.20]

    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    base, meta = prepare_fn(tmp_path)
    total_demand = float(meta["od_matrix"].sum())
    n_links = meta["n_links"]
    n_zones = meta["n_zones"]
    node_coords = meta["nodes"]
    link_attrs = meta["link_attrs"]
    ref = meta.get("ref_flows", {})

    figs: list[go.Figure | None] = []
    descriptions: list[str] = []

    # --- 0. Network topology map ---
    logger.info("[%s] Building topology map...", network_name)
    _add_topology_section(
        figs, descriptions, network_name, node_coords, link_attrs,
        n_links, n_zones, total_demand,
    )

    # --- 1. Demand scaling sweep ---
    logger.info(
        "[%s] Running demand sweep (%d scales)...",
        network_name, len(sweep_scales),
    )
    _add_sweep_section(
        figs, descriptions, base, meta, copy_fn, run_fn,
        sweep_scales, total_demand, n_links, max_iter,
    )

    # --- 2. FW vs MSA convergence ---
    logger.info(
        "[%s] Running convergence comparison (FW + MSA, %d iters)...",
        network_name, max_iter,
    )
    result_fw = _add_convergence_section(
        figs, descriptions, base, meta, copy_fn, prepare_fn, run_fn,
        tmp_path, detail_scale, total_demand, max_iter,
    )

    # --- 3. Flow and TT correlation ---
    state = result_fw.network_state
    if ref:
        logger.info("[%s] Building correlation plots...", network_name)
        _add_correlation_section(
            figs, descriptions, state, ref, link_attrs, detail_scale,
        )

    # --- 4. V/C scatter ---
    logger.info(
        "[%s] Running V/C scatter (%d scales)...",
        network_name, len(vc_scales),
    )
    _add_vc_section(
        figs, descriptions, base, meta, copy_fn, run_fn,
        link_attrs, vc_scales, max_iter, tmp_path,
    )

    # --- 5. Link state table (at end — large for big networks) ---
    logger.info("[%s] Building link state table...", network_name)
    _add_link_table_section(
        figs, descriptions, state, link_attrs, detail_scale, total_demand,
    )

    # Compose intro
    lane_counts = [a["n_lanes"] for a in link_attrs.values()]
    speed_set = sorted(set(round(a["ff_speed_kmh"]) for a in link_attrs.values()))
    auto_intro = (
        f"<p>Validation of density-based traffic assignment on the "
        f"<b>{network_name}</b> benchmark "
        f"({len(node_coords)} nodes, {n_links} links, {n_zones} zones). "
        f"Total TNTP demand: {total_demand:,.0f} vph. "
        f"Lanes: {min(lane_counts)}&ndash;{max(lane_counts)}. "
        f"Freeflow speeds: {', '.join(str(s) for s in speed_set)} km/h. "
        f"VDF: bi-parabolic MFD (k<sub>j</sub>=200 veh/km/lane). "
        f"Detail analysis at {detail_scale:.0%} demand "
        f"({total_demand * detail_scale:,.0f} vph).</p>"
    )
    if intro_html:
        auto_intro += intro_html

    _write_combined_report(
        title=f"{network_name} Validation",
        intro=auto_intro,
        figures=figs,
        descriptions=descriptions,
        path=Path(output_path),
    )
    return Path(output_path)


# ---------------------------------------------------------------------------
# Report section builders (private)
# ---------------------------------------------------------------------------

def _add_topology_section(figs, descriptions, name, nodes, link_attrs,
                          n_links, n_zones, total_demand):
    """Section 0: network topology map."""
    fig = go.Figure()
    for (u, v), attrs in link_attrs.items():
        if u not in nodes or v not in nodes:
            continue
        x0, y0 = nodes[u]
        x1, y1 = nodes[v]
        lanes = attrs["n_lanes"]
        color = "#D32F2F" if lanes >= 3 else "#2196F3"
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=color, width=max(1, lanes * 1.2)),
            hoverinfo="text",
            hovertext=f"{u}&rarr;{v}: {lanes}L, {attrs['ff_speed_kmh']:.0f} km/h",
            showlegend=False,
        ))

    xs = [nodes[n][0] for n in sorted(nodes)]
    ys = [nodes[n][1] for n in sorted(nodes)]
    labels = [str(n) for n in sorted(nodes)]
    show_labels = len(nodes) <= 50
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text" if show_labels else "markers",
        marker=dict(
            size=16 if show_labels else 4,
            color="#4CAF50",
            line=dict(width=1.5, color="white") if show_labels else dict(width=0),
        ),
        text=labels if show_labels else None,
        textfont=dict(size=9, color="white") if show_labels else None,
        textposition="middle center" if show_labels else None,
        hovertext=[f"Node {n}" for n in sorted(nodes)],
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=f"{name} Network ({len(nodes)} nodes, {n_links} links)",
        xaxis=dict(title="Longitude", scaleanchor="y", fixedrange=True),
        yaxis=dict(title="Latitude", fixedrange=True),
        template="plotly_white",
        height=500,
    )
    figs.append(fig)

    n_major = sum(1 for a in link_attrs.values() if a["n_lanes"] >= 3)
    n_minor = sum(1 for a in link_attrs.values() if a["n_lanes"] < 3)
    descriptions.append(
        f"<h2>Network Topology</h2>"
        f"<p>{name}: {len(nodes)} nodes, {n_links} directed links, {n_zones} zones. "
        f'<span style="color:#D32F2F"><b>{n_major} links</b></span> '
        f"with &ge;3 lanes, "
        f'<span style="color:#2196F3"><b>{n_minor} links</b></span> '
        f"with &lt;3 lanes. "
        f"Total TNTP demand: {total_demand:,.0f} vph.</p>"
    )


def _add_sweep_section(figs, descriptions, base, meta, copy_fn, run_fn,
                        scales, total_demand, n_links, max_iter):
    """Section 1: demand scaling sweep (oversat, speed, TSTT)."""
    sweep_demand = []
    sweep_oversat = []
    sweep_mean_speed = []
    sweep_tstt = []

    for scale in scales:
        logger.info("  Sweep scale %.0f%%...", scale * 100)
        run_base = copy_fn(base, Path(base).parent.parent / f"sweep_{scale:.2f}")
        result = run_fn(run_base, meta, max_iter, "fw", scale)
        state = result.network_state
        last = result.iteration_log[-1]
        speeds = state.speed_kmh[:state.n_edges]
        ff = state.freeflow_kmh[:state.n_edges]

        sweep_demand.append(total_demand * scale)
        sweep_oversat.append(last.n_oversaturated)
        sweep_mean_speed.append(float(np.mean(speeds / ff)))
        sweep_tstt.append(last.tstt)

    labels = [f"{s:.0%}" for s in scales]

    # 1a: Oversaturated links
    fig_oversat = go.Figure()
    fig_oversat.add_trace(go.Bar(
        x=labels, y=sweep_oversat,
        marker_color=[
            "#4CAF50" if o < 5 else "#FF9800" if o < 40 else "#D32F2F"
            for o in sweep_oversat
        ],
        hovertext=[
            f"{d:,.0f} vph &rarr; {o}/{n_links} links oversat"
            for d, o in zip(sweep_demand, sweep_oversat)
        ],
        hoverinfo="text",
    ))
    fig_oversat.update_layout(
        title="Oversaturated Links vs Demand Scale",
        xaxis_title="Demand Scale (% of TNTP)",
        yaxis_title="Links at Jam Density",
        template="plotly_white",
        yaxis=dict(fixedrange=True), xaxis=dict(fixedrange=True),
        showlegend=False,
    )
    figs.append(fig_oversat)
    descriptions.append(
        "<h2>Demand Scaling: Oversaturation</h2>"
        "<p>Number of links reaching jam density as demand increases.</p>"
    )

    # 1b: Mean speed ratio
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
        "drops to the minimum (jam).</p>"
    )

    # 1c: TSTT
    fig_tstt = go.Figure()
    fig_tstt.add_trace(go.Scatter(
        x=[total_demand * s for s in scales],
        y=sweep_tstt,
        mode="lines+markers",
        line=dict(color="#D32F2F", width=2.5),
        marker=dict(size=8),
        hovertext=[f"{s:.0%}: TSTT={t:,.0f}" for s, t in zip(scales, sweep_tstt)],
        hoverinfo="text",
    ))
    fig_tstt.update_layout(
        title="Total System Travel Time vs Demand",
        xaxis_title="Total Demand (vph)",
        yaxis_title="TSTT (veh-seconds)",
        yaxis_type="log",
        template="plotly_white",
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True, dtick=1, tickformat=".0e"),
    )
    figs.append(fig_tstt)
    descriptions.append(
        "<h2>Demand Scaling: TSTT</h2>"
        "<p>Total system travel time rises exponentially as demand approaches "
        "physical capacity.</p>"
    )


def _add_convergence_section(figs, descriptions, base, meta, copy_fn,
                              prepare_fn, run_fn, tmp_path,
                              detail_scale, total_demand, max_iter):
    """Section 2: FW vs MSA convergence. Returns FW result."""
    logger.info("  Running FW at %.0f%% demand...", detail_scale * 100)
    base_fw = copy_fn(base, tmp_path / "fw_detail")
    result_fw = run_fn(base_fw, meta, max_iter, "fw", detail_scale)

    logger.info("  Running MSA at %.0f%% demand...", detail_scale * 100)
    base_msa, meta_msa = prepare_fn(tmp_path / "msa")
    result_msa = run_fn(base_msa, meta_msa, max_iter, "msa", detail_scale)

    iters_fw = [r.iteration for r in result_fw.iteration_log]
    gap_fw = [r.relative_gap for r in result_fw.iteration_log]
    iters_msa = [r.iteration for r in result_msa.iteration_log]
    gap_msa = [r.relative_gap for r in result_msa.iteration_log]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters_fw, y=[abs(g) if g != 0 else None for g in gap_fw],
        mode="lines+markers", name="Frank-Wolfe",
        line=dict(color="#D32F2F", width=2.5), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=iters_msa, y=[abs(g) if g != 0 else None for g in gap_msa],
        mode="lines+markers", name="MSA (1/n)",
        line=dict(color="#1565C0", width=1.5, dash="dash"), marker=dict(size=3),
    ))
    fig.update_layout(
        title=f"Wardrop Gap: FW vs MSA ({detail_scale:.0%} Demand)",
        xaxis_title="Iteration", yaxis_title="|Relative Gap|",
        yaxis_type="log",
        template="plotly_white",
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True, dtick=1, tickformat=".0e"),
    )
    figs.append(fig)

    fw_final = result_fw.iteration_log[-1]
    msa_final = result_msa.iteration_log[-1]
    descriptions.append(
        f"<h2>Convergence at {detail_scale:.0%} Demand</h2>"
        f"<p>FW gap: {fw_final.relative_gap:.4f}, "
        f"MSA gap: {msa_final.relative_gap:.4f} ({max_iter} iterations). "
        f"Demand: {total_demand * detail_scale:,.0f} vph "
        f"({detail_scale:.0%} of TNTP).</p>"
    )

    return result_fw


def _add_link_table_section(figs, descriptions, state, link_attrs,
                             detail_scale, total_demand):
    """Section 3: per-link state table."""
    rows = []
    for i in range(state.n_edges):
        key = (int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1]))
        la = link_attrs.get(key, {})
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

    all_vc = [r[9] for r in rows]
    all_speeds = [r[4] for r in rows]
    finite_ratios = [r[8] / r[7] for r in rows if r[7] > 0 and r[8] < float("inf")]
    n_congested = sum(1 for vc in all_vc if vc > 0.6)

    figs.append(None)
    descriptions.append(
        f"<h2>Link State at {detail_scale:.0%} Demand</h2>"
        f"<p>{total_demand * detail_scale:,.0f} vph ({state.n_edges} links). "
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


def _add_correlation_section(figs, descriptions, state, ref, link_attrs,
                              detail_scale):
    """Section 4: flow and TT correlation vs BPR reference."""
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
        la = link_attrs.get(key, {})
        dist_m = la.get("distance_m", 0)
        v = state.speed_kmh[i]
        our_flow = state.flow_vph[i]
        our_tt = dist_m / 1000 / v * 60 if v > 0 else float("inf")

        mfd_flows.append(our_flow)
        bpr_flows.append(bpr_flow)
        mfd_tt.append(our_tt)
        bpr_tt.append(bpr_cost)
        link_labels.append(f"{key[0]}&rarr;{key[1]}")

    # 4a: Flow scatter
    fig_flow = go.Figure()
    fig_flow.add_trace(go.Scatter(
        x=bpr_flows, y=mfd_flows, mode="markers",
        marker=dict(size=6, color="#1565C0", opacity=0.7),
        hovertext=link_labels, hoverinfo="text",
    ))
    flow_rho = 0.0
    if len(mfd_flows) >= 3:
        flow_rho, _ = spearmanr(bpr_flows, mfd_flows)
    fig_flow.update_layout(
        title=f"Link Flow: MFD ({detail_scale:.0%}) vs BPR (100%) — ρ={flow_rho:.3f}",
        xaxis_title="BPR Equilibrium Flow (vph, 100% demand)",
        yaxis_title=f"MFD Flow (vph, {detail_scale:.0%} demand)",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig_flow)
    descriptions.append(
        "<h2>Flow Correlation: MFD vs BPR</h2>"
        "<p>Scatter of per-link flow: our density-based MFD assignment "
        f"({detail_scale:.0%} demand, {state.n_edges} links) vs published BPR equilibrium "
        "(100% demand). The <b>rank correlation</b> (Spearman &rho;) measures whether "
        "the same links carry relatively more or less traffic under both models. "
        f"&rho; = <b>{flow_rho:.3f}</b> ({len(mfd_flows)} matched links).</p>"
    )

    # 4b: TT scatter (exclude gridlocked)
    finite_mask = [t < float("inf") for t in mfd_tt]
    tt_bpr_f = [b for b, m in zip(bpr_tt, finite_mask) if m]
    tt_mfd_f = [t for t, m in zip(mfd_tt, finite_mask) if m]
    tt_labels_f = [l for l, m in zip(link_labels, finite_mask) if m]
    n_gridlocked = sum(1 for m in finite_mask if not m)

    fig_tt = go.Figure()
    fig_tt.add_trace(go.Scatter(
        x=tt_bpr_f, y=tt_mfd_f, mode="markers",
        marker=dict(size=6, color="#D32F2F", opacity=0.7),
        hovertext=tt_labels_f, hoverinfo="text",
    ))
    if tt_bpr_f and tt_mfd_f:
        tt_max = max(max(tt_bpr_f), max(tt_mfd_f)) * 1.1
        fig_tt.add_shape(
            type="line", x0=0, x1=tt_max, y0=0, y1=tt_max,
            line=dict(color="#999", dash="dash", width=1),
        )
    tt_rho = 0.0
    if len(tt_mfd_f) >= 3:
        tt_rho, _ = spearmanr(tt_bpr_f, tt_mfd_f)
    fig_tt.update_layout(
        title=f"Link Travel Time: MFD vs BPR — ρ={tt_rho:.3f}",
        xaxis_title="BPR Equilibrium Travel Time (min)",
        yaxis_title=f"MFD Travel Time (min, {detail_scale:.0%} demand)",
        template="plotly_white",
        xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True),
    )
    figs.append(fig_tt)
    gridlock_note = (
        f" ({n_gridlocked} gridlocked link{'s' if n_gridlocked != 1 else ''} excluded.)"
        if n_gridlocked > 0 else ""
    )
    descriptions.append(
        "<h2>Travel Time Correlation: MFD vs BPR</h2>"
        f"<p>Per-link travel time: MFD at {detail_scale:.0%} demand ({len(tt_mfd_f)} "
        "links) vs BPR at 100% demand. Dashed line = y&thinsp;=&thinsp;x. "
        f"Spearman &rho; = <b>{tt_rho:.3f}</b>.{gridlock_note}</p>"
    )


def _add_vc_section(figs, descriptions, base, meta, copy_fn, run_fn,
                     link_attrs, scales, max_iter, tmp_path):
    """Section 5: V/C scatter across demand levels."""
    fig = go.Figure()
    colors = ["#4CAF50", "#1565C0", "#FF6F00", "#D32F2F",
              "#9C27B0", "#00BCD4", "#795548", "#607D8B"]
    max_flow = 1

    for sc, col in zip(scales, colors):
        logger.info("  V/C scale %.0f%%...", sc * 100)
        vc_base = copy_fn(base, tmp_path / f"vc_{sc:.2f}")
        res = run_fn(vc_base, meta, max_iter, "fw", sc)
        st = res.network_state
        vc_list = []
        flow_list = []
        hover_list = []
        for i in range(st.n_edges):
            k = (int(st.edge_ids[i, 0]), int(st.edge_ids[i, 1]))
            vf_i = st.freeflow_kmh[i]
            kj_i = st.jam_density[i]
            qc_i = vf_i * kj_i / 6
            vc_i = st.flow_vph[i] / qc_i if qc_i > 0 else 0
            vc_list.append(vc_i)
            flow_list.append(st.flow_vph[i])
            hover_list.append(f"{k[0]}&rarr;{k[1]}: V/C={vc_i:.2f}")
        max_flow = max(max_flow, max(flow_list) if flow_list else 1)
        fig.add_trace(go.Scatter(
            x=flow_list, y=vc_list, mode="markers",
            marker=dict(size=5, color=col, opacity=0.6),
            name=f"{sc:.0%} demand",
            hovertext=hover_list, hoverinfo="text",
        ))

    fig.add_shape(
        type="line", x0=0, x1=max_flow * 1.1,
        y0=1, y1=1, line=dict(color="#D32F2F", dash="dash", width=1),
    )
    fig.update_layout(
        title="Link V/C Ratio at Different Demand Levels",
        xaxis_title="Link Flow (vph)",
        yaxis_title="Volume / Capacity",
        template="plotly_white",
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True, range=[0, 1.5]),
    )
    figs.append(fig)
    descriptions.append(
        "<h2>V/C by Demand Level</h2>"
        "<p>Each point is one link. V/C = 1.0 is the MFD physical capacity "
        "ceiling.</p>"
    )
