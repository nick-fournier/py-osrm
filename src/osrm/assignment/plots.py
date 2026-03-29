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
    fig.update_yaxes(title_text="|q_out - q_in|", row=1, col=2)

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


def vdf_theory(output_dir: str = "docs/plots") -> list[Path]:
    """Generate all VDF theory validation plots.

    Returns list of saved HTML file paths.
    """
    d = Path(output_dir)
    paths = [
        vdf_speed_density(path=str(d / "vdf_speed_density.html")),
        vdf_flow_density(path=str(d / "vdf_flow_density.html")),
        vdf_inverse_accuracy(path=str(d / "vdf_inverse_accuracy.html")),
        vdf_multi_class(path=str(d / "vdf_multi_class.html")),
    ]
    return [Path(str(d / f)) for f in [
        "vdf_speed_density.html",
        "vdf_flow_density.html",
        "vdf_inverse_accuracy.html",
        "vdf_multi_class.html",
    ]]


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
