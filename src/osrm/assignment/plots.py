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


def _freeflow_branch_traveltime(
    q: np.ndarray,
    q_c: float,
    v_f: float,
    link_length_km: float,
) -> np.ndarray:
    """Travel time on the uncongested branch as a function of demand rate."""
    ratio = np.clip(q / q_c, 0.0, 1.0)
    return 120.0 * link_length_km / (v_f * (1.0 + np.sqrt(1.0 - ratio)))


def _queue_delay_surrogate_traveltime(
    q: np.ndarray,
    q_c: float,
    v_f: float,
    link_length_km: float,
    analysis_period_hr: float,
) -> np.ndarray:
    """Single-valued static surrogate: running time + average point-queue delay."""
    t_free = _freeflow_branch_traveltime(q, q_c, v_f, link_length_km)
    t_c = 120.0 * link_length_km / v_f
    over_capacity = np.maximum(q / q_c - 1.0, 0.0)
    queue_delay_min = 30.0 * analysis_period_hr * over_capacity
    return np.where(q <= q_c, t_free, t_c + queue_delay_min)


def _steep_tail_surrogate_traveltime(
    q: np.ndarray,
    q_c: float,
    v_f: float,
    link_length_km: float,
    tail_power: float,
) -> np.ndarray:
    """Single-valued static surrogate: uncongested branch + large-power tail."""
    t_free = _freeflow_branch_traveltime(q, q_c, v_f, link_length_km)
    t_c = 120.0 * link_length_km / v_f
    return np.where(q <= q_c, t_free, t_c * np.power(q / q_c, tail_power))


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


    return _save_or_show(fig, path)


def vdf_speed_flow(
    v_f: float = 60.0,
    k_j: float = 150.0,
    vdf: BiParabolicVDF | None = None,
    path: Optional[str] = None,
) -> go.Figure:
    """Plot the speed–flow relationship (backward-bending curve).

    This is the parametric curve {q(k), v(k)} as k sweeps from 0 to k_j.
    The characteristic backward bend shows that in the congested regime,
    both speed and flow decrease — the hallmark of traffic breakdown.
    """
    vdf = vdf or BiParabolicVDF()
    k_c = vdf.kc_ratio * k_j
    q_c = v_f * k_c / 2.0
    v_c = v_f / 2.0

    k = np.linspace(0, k_j, 500)
    v_f_arr = np.full_like(k, v_f)
    k_j_arr = np.full_like(k, k_j)
    v = vdf.density_to_speed(k, v_f_arr, k_j_arr)
    q = vdf.density_to_flow(k, v_f_arr, k_j_arr)

    mask_unc = k <= k_c
    mask_con = k > k_c

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=q[mask_unc], y=v[mask_unc],
        mode="lines", name="Uncongested",
        line=dict(color="#2196F3", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=q[mask_con], y=v[mask_con],
        mode="lines", name="Congested",
        line=dict(color="#F44336", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[q_c], y=[v_c],
        mode="markers+text", name=f"Capacity (q_c={q_c:.0f}, v_c={v_c:.0f})",
        marker=dict(size=12, color="#FF9800", symbol="diamond"),
        text=[f"q_c={q_c:.0f}"],
        textposition="top left",
    ))

    fig.update_layout(
        title=f"Bi-Parabolic Speed–Flow (v_f={v_f}, k_j={k_j})",
        xaxis_title="Flow q (veh/hr)",
        yaxis_title="Speed v (km/h)",
        template="plotly_white",
        legend=dict(x=0.6, y=0.95),
    )
    return _save_or_show(fig, path)


def vdf_flow_traveltime(
    v_f: float = 60.0,
    k_j: float = 150.0,
    link_length_km: float = 1.0,
    analysis_period_hr: float = 1.0,
    surrogate_max_ratio: float = 1.2,
    tail_power: float = 20.0,
    vdf: BiParabolicVDF | None = None,
    path: Optional[str] = None,
) -> go.Figure:
    """Plot the flow–travel-time relationship (MFD branches only).

    Travel time t = L / v(k) for a link of length *link_length_km*.
    This is the cost function that assignment directly optimizes:
    on the uncongested branch, travel time increases with flow;
    at breakdown, both flow drops and travel time spikes.
    """
    vdf = vdf or BiParabolicVDF()
    k_c = vdf.kc_ratio * k_j
    q_c = v_f * k_c / 2.0

    k = np.linspace(0.001, k_j, 500)
    v_f_arr = np.full_like(k, v_f)
    k_j_arr = np.full_like(k, k_j)
    v = vdf.density_to_speed(k, v_f_arr, k_j_arr)
    q = vdf.density_to_flow(k, v_f_arr, k_j_arr)
    tt = link_length_km / (v / 60.0)  # minutes

    t_ff = link_length_km / (v_f / 60.0)
    t_c = link_length_km / (v_f / 2.0 / 60.0)

    mask_unc = k <= k_c
    mask_con = k > k_c

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=q[mask_unc], y=tt[mask_unc],
        mode="lines", name="Uncongested MFD branch",
        line=dict(color="#2196F3", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=q[mask_con], y=tt[mask_con],
        mode="lines", name="Congested MFD branch",
        line=dict(color="#F44336", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[q_c], y=[t_c],
        mode="markers+text", name=f"Capacity (q_c={q_c:.0f})",
        marker=dict(size=12, color="#FF9800", symbol="diamond"),
        text=[f"t_c={t_c:.2f} min"],
        textposition="top left",
    ))
    fig.add_hline(y=t_ff, line_dash="dot", line_color="gray",
                  annotation_text=f"t_ff={t_ff:.2f} min",
                  annotation_position="bottom right")

    fig.update_layout(
        title=f"Flow–Travel Time (L={link_length_km} km, v_f={v_f}, k_j={k_j})",
        xaxis_title="Flow q (veh/hr)",
        yaxis_title="Travel time (min)",
        template="plotly_white",
        legend=dict(x=0.05, y=0.95),
    )
    fig.update_xaxes(range=[0.0, q_c * surrogate_max_ratio])
    fig.update_yaxes(range=[0.0, max(10 * t_c, 30)])
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


def vdf_inverse_mfd(
    v_f: float = 60.0,
    k_j: float = 150.0,
    vdf: BiParabolicVDF | None = None,
    path: Optional[str] = None,
) -> go.Figure:
    """Plot the inverse fundamental diagram: q → k for both branches.

    Given flow q, the two-branch inversion is:

    Free-flow:   k(q) = k_c (1 − √(1 − 2q / (v_f k_c)))
    Congested:   k(q) = k_c + (k_j − k_c) √(1 − 2q / (v_f k_c))

    Both branches exist for 0 ≤ q ≤ q_c.  Demand volumes above q_c
    have no solution — the link is over capacity.
    """
    vdf = vdf or BiParabolicVDF()
    k_c = vdf.kc_ratio * k_j
    q_c = v_f * k_c / 2.0

    q = np.linspace(0, q_c * 0.999, 500)
    discriminant = np.sqrt(np.maximum(1.0 - 2.0 * q / (v_f * k_c), 0.0))

    k_freeflow = k_c * (1.0 - discriminant)
    k_congested = k_c + (k_j - k_c) * discriminant

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=q, y=k_freeflow,
        mode="lines", name="Free-flow branch",
        line=dict(color="#2196F3", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=q, y=k_congested,
        mode="lines", name="Congested branch",
        line=dict(color="#F44336", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[q_c], y=[k_c],
        mode="markers+text", name=f"Capacity (q_c={q_c:.0f}, k_c={k_c:.0f})",
        marker=dict(size=12, color="#FF9800", symbol="diamond"),
        text=[f"q_c={q_c:.0f}"],
        textposition="top left",
    ))
    fig.add_hline(y=k_j, line_dash="dot", line_color="gray",
                  annotation_text=f"k_j={k_j:.0f}", annotation_position="bottom right")

    fig.update_layout(
        title=f"Inverse MFD: q → k (v_f={v_f}, k_j={k_j})",
        xaxis_title="Flow q (veh/hr)",
        yaxis_title="Density k (veh/km)",
        template="plotly_white",
        legend=dict(x=0.05, y=0.95),
    )
    return _save_or_show(fig, path)


def _vdf_extended_cost_chain(
    v_f: float = 60.0,
    k_j: float = 150.0,
    link_length_km: float = 1.0,
    max_demand_ratio: float = 5.0,
    analysis_period_hr: float = 1.0,
    tail_power: float = 20.0,
    vdf: BiParabolicVDF | None = None,
) -> go.Figure:
    """Three-panel plot of the extended q→k→v→t cost chain for oversaturated demand.

    Left:   Extended q → k mapping (uncongested inverse + asymptotic extension)
            with MFD congested-branch inverse as reference
    Center: Speed–flow (MFD parametric + extended demand mapping)
    Right:  Flow–travel time comparing all surrogate cost functions:
            MFD cost chain, BPR, queue-delay, steep-tail, and MFD parametric
    """
    vdf = vdf or BiParabolicVDF()
    k_c = vdf.kc_ratio * k_j
    q_c = v_f * k_c / 2.0
    v_c = v_f / 2.0

    # --- Demand range: 0 to max_demand_ratio × q_c ---
    q_demand = np.linspace(0.001, q_c * max_demand_ratio, 1000)
    v_f_arr = np.full_like(q_demand, v_f)
    k_j_arr = np.full_like(q_demand, k_j)

    # Extended mapping: demand → density → speed → travel time
    k_ext = vdf.demand_to_density(q_demand, v_f_arr, k_j_arr)
    v_ext = vdf.density_to_speed(k_ext, v_f_arr, k_j_arr)
    tt_ext = link_length_km / v_ext * 60.0  # minutes

    # Parametric MFD (for the backward-bending reference)
    k_param = np.linspace(0.001, k_j * 0.999, 500)
    v_param = vdf.density_to_speed(k_param, np.full_like(k_param, v_f), np.full_like(k_param, k_j))
    q_param = vdf.density_to_flow(k_param, np.full_like(k_param, v_f), np.full_like(k_param, k_j))
    tt_param = link_length_km / v_param * 60.0

    # MFD congested-branch inverse for q→k panel (maps q→k on the congested side)
    q_inv = np.linspace(0.001, q_c * 0.999, 300)
    disc = np.sqrt(np.maximum(1.0 - 2.0 * q_inv / (v_f * k_c), 0.0))
    k_congested_inv = k_c + (k_j - k_c) * disc

    # BPR comparison: t = t0 * [1 + 0.15 * (q/qc)^4]
    t0_min = link_length_km / v_f * 60.0
    tt_bpr = t0_min * (1.0 + 0.15 * (q_demand / q_c) ** 4)

    # Queue-delay and steep-tail surrogates
    tt_queue = _queue_delay_surrogate_traveltime(
        q_demand, q_c, v_f, link_length_km, analysis_period_hr
    )
    tt_tail = _steep_tail_surrogate_traveltime(
        q_demand, q_c, v_f, link_length_km, tail_power
    )

    # Masks for uncongested / oversaturated
    mask_under = q_demand <= q_c
    mask_over = q_demand > q_c

    # Parametric MFD masks
    mask_param_unc = k_param <= k_c
    mask_param_con = k_param > k_c

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "q → k Mapping",
            "Speed vs Demand",
            "Travel Time vs Demand (all surrogates)",
        ],
        horizontal_spacing=0.08,
    )

    # === Panel 1: q → k mapping ===
    # MFD congested-branch inverse (reference)
    fig.add_trace(go.Scatter(
        x=q_inv, y=k_congested_inv,
        mode="lines", name="MFD congested inverse",
        line=dict(color="#999", width=1.5, dash="dot"),
        legendgroup="mfd_ref",
    ), row=1, col=1)
    # Extended mapping
    fig.add_trace(go.Scatter(
        x=q_demand[mask_under], y=k_ext[mask_under],
        mode="lines", name="Uncongested inverse",
        line=dict(color="#2196F3", width=2.5),
        legendgroup="ext",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=q_demand[mask_over], y=k_ext[mask_over],
        mode="lines", name="Asymptotic extension",
        line=dict(color="#F44336", width=2.5),
        legendgroup="ext",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[q_c], y=[k_c],
        mode="markers", showlegend=False,
        marker=dict(size=10, color="#FF9800", symbol="diamond"),
    ), row=1, col=1)
    fig.add_hline(y=k_j, line_dash="dot", line_color="gray", row=1, col=1)
    fig.add_hline(y=k_c, line_dash="dot", line_color="#ddd", row=1, col=1)

    # === Panel 2: Speed vs demand ===
    # MFD parametric (backward-bending reference)
    fig.add_trace(go.Scatter(
        x=q_param[mask_param_unc], y=v_param[mask_param_unc],
        mode="lines", name="MFD parametric",
        line=dict(color="#999", width=1.5, dash="dot"),
        legendgroup="mfd_ref", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=q_param[mask_param_con], y=v_param[mask_param_con],
        mode="lines",
        line=dict(color="#999", width=1.5, dash="dot"),
        legendgroup="mfd_ref", showlegend=False,
    ), row=1, col=2)
    # Extended cost chain
    fig.add_trace(go.Scatter(
        x=q_demand, y=v_ext,
        mode="lines", name="MFD cost chain v(q)",
        line=dict(color="#4CAF50", width=3),
        legendgroup="chain",
    ), row=1, col=2)

    # === Panel 3: Travel time vs demand (all surrogates) ===
    # MFD parametric (backward-bending reference)
    fig.add_trace(go.Scatter(
        x=q_param[mask_param_unc], y=tt_param[mask_param_unc],
        mode="lines",
        line=dict(color="#999", width=1.5, dash="dot"),
        legendgroup="mfd_ref", showlegend=False,
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=q_param[mask_param_con], y=tt_param[mask_param_con],
        mode="lines",
        line=dict(color="#999", width=1.5, dash="dot"),
        legendgroup="mfd_ref", showlegend=False,
    ), row=1, col=3)
    # Extended MFD cost chain
    fig.add_trace(go.Scatter(
        x=q_demand, y=tt_ext,
        mode="lines", name="MFD cost chain t(q)",
        line=dict(color="#4CAF50", width=3),
        legendgroup="chain",
    ), row=1, col=3)
    # BPR
    fig.add_trace(go.Scatter(
        x=q_demand, y=tt_bpr,
        mode="lines", name="BPR(0.15, 4)",
        line=dict(color="#9C27B0", width=2.5, dash="dash"),
        legendgroup="surr",
    ), row=1, col=3)
    # Queue-delay
    fig.add_trace(go.Scatter(
        x=q_demand, y=tt_queue,
        mode="lines", name="Queue-delay",
        line=dict(color="#FF9800", width=2.5, dash="dashdot"),
        legendgroup="surr",
    ), row=1, col=3)
    # Steep-tail
    fig.add_trace(go.Scatter(
        x=q_demand, y=tt_tail,
        mode="lines", name=f"Steep-tail (power {tail_power:.0f})",
        line=dict(color="#00BCD4", width=2.5, dash="dot"),
        legendgroup="surr",
    ), row=1, col=3)
    # Capacity marker
    fig.add_trace(go.Scatter(
        x=[q_c], y=[tt_ext[np.argmin(np.abs(q_demand - q_c))]],
        mode="markers", showlegend=False,
        marker=dict(size=8, color="#FF9800", symbol="diamond"),
    ), row=1, col=3)

    fig.update_xaxes(title_text="Demand q (veh/hr)", row=1, col=1)
    fig.update_xaxes(title_text="Demand q (veh/hr)", row=1, col=2)
    fig.update_xaxes(title_text="Demand q (veh/hr)", row=1, col=3)
    fig.update_yaxes(title_text="Density k (veh/km)", row=1, col=1)
    fig.update_yaxes(title_text="Speed v (km/h)", row=1, col=2)
    fig.update_yaxes(title_text="Travel time (min)", range=[0, 100], row=1, col=3)

    fig.update_layout(
        title=f"Extended Cost Chain: q → k → v → t  (v_f={v_f}, k_j={k_j}, L={link_length_km}km)",
        template="plotly_white",
        height=450,
        showlegend=True,
    )
    return fig


def vdf_theory(output_dir: str = "plots") -> Path:
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
    analysis_period_hr = 1.0
    tail_power = 20.0

    figs = [
        vdf_speed_density(v_f=v_f, k_j=k_j, vdf=vdf),
        vdf_flow_density(v_f=v_f, k_j=k_j, vdf=vdf),
        vdf_speed_flow(v_f=v_f, k_j=k_j, vdf=vdf),
        vdf_flow_traveltime(
            v_f=v_f,
            k_j=k_j,
            analysis_period_hr=analysis_period_hr,
            tail_power=tail_power,
            vdf=vdf,
        ),
        None,
        vdf_inverse_mfd(v_f=v_f, k_j=k_j, vdf=vdf),
        _vdf_near_jam_detail(v_f=v_f, k_j=k_j, vdf=vdf),
        vdf_inverse_accuracy(v_f=v_f, k_j=k_j, vdf=vdf),
        vdf_multi_class(vdf=vdf),
        _vdf_extended_cost_chain(
            v_f=v_f, k_j=k_j, vdf=vdf,
            analysis_period_hr=analysis_period_hr,
            tail_power=tail_power,
        ),
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

        f"""<h2>3. Speed–Flow Relationship</h2>
        <p>The characteristic backward-bending speed–flow curve, derived parametrically
        from {{q(k), v(k)}} as density sweeps from 0 to k<sub>j</sub>. On the
        <b>uncongested branch</b> (blue), increasing flow reduces speed — the familiar
        "more cars, slower travel" intuition. At capacity
        q<sub>c</sub> = {q_c:.0f} veh/hr, speed is v<sub>c</sub> = v<sub>f</sub>/2 = {v_f/2:.0f} km/h.
        Beyond this point the curve bends backward: on the <b>congested branch</b> (red),
        <em>both</em> speed and flow decrease — the hallmark of traffic breakdown.
        This is the relationship most directly observed by highway sensors and
        loop detectors.</p>""",

        f"""<h2>4. Flow–Travel Time (Cost Function)</h2>
        <p>Travel time $t = L / v$ for a 1 km link, plotted against flow. This is
        the link cost function that assignment directly optimizes. On the
        <b>uncongested branch</b> (blue), travel time rises monotonically from
        the freeflow time as flow increases toward capacity — the classical
        "congestion penalty." At capacity breakdown, the curve bends backward:
        on the <b>congested branch</b> (red), flow drops while travel time continues
        to spike. Near jam density, travel time diverges.
        This backward bend is why BPR-style $t(V)$ cost functions are monotonic
        approximations — they avoid the multi-valued regime. Our density-based
        VDF handles both branches natively via $t = L / v(k)$, where $k$ is
        always single-valued. See §10 for the monotone cost extensions that
        resolve the backward bend for assignment.</p>""",

        f"""<h2>5. Queue Delay and Steep-Tail Static Surrogates</h2>
        <p>The backward-bending MFD is physically meaningful, but static assignment
        needs a single-valued cost curve in demand space. A queue-delay surrogate
        keeps the <b>running time</b> on the uncongested branch up to capacity and,
        for $q &gt; q_c$, adds the average delay from a point queue that forms when
        arrivals exceed discharge:</p>
        <p>$$t^{{queue}}(q) =
        \\begin{{cases}}
        t_{{run}}(q), & q \\leq q_c \\\\
        t_c + \\dfrac{{H}}{{2}}\\left(\\dfrac{{q}}{{q_c}} - 1\\right), & q &gt; q_c
        \\end{{cases}}$$</p>
        <p>where $H$ is the analysis period and $t_c = 2L/v_f$ is the running time
        at capacity. In this report the overlay uses $H = {analysis_period_hr:.0f}$ hour, so the queue
        penalty grows by 30 minutes for each additional capacity's worth of demand.
        The purple comparator is the proposed steep-tail extension
        $t^{{tail}}(q) = t_c (q/q_c)^{{{tail_power:.0f}}}$ above capacity. It is numerically simple
        and monotone, but unlike queue delay it does not correspond to an explicit
        storage or discharge process.</p>""",

        f"""<h2>6. Inverse MFD: Flow → Density</h2>
        <p>The closed-form inversion of the MFD. Given flow $q$, density on each branch is:</p>
        <p><b>Free-flow:</b> &emsp; $k(q) = k_c \\left(1 - \\sqrt{{1 - \\dfrac{{2q}}{{v_f k_c}}}}\\right)$</p>
        <p><b>Congested:</b> &emsp; $k(q) = k_c + (k_j - k_c)\\sqrt{{1 - \\dfrac{{2q}}{{v_f k_c}}}}$</p>
        <p>Both branches exist only for $0 \\leq q \\leq q_c = {q_c:.0f}$ veh/hr.
        At $q = q_c$ they meet at $k_c = {k_c:.0f}$. At $q = 0$, the free-flow branch
        gives $k = 0$ while the congested branch gives $k = k_j = {k_j:.0f}$.
        Demand volumes exceeding $q_c$ are out of range — the link is over capacity
        and the inversion has no real solution, which is why convergence blending
        operates in volume space with a constant-factor density derivation.</p>""",

        f"""<h2>7. Near-Jam Behaviour (k → k<sub>j</sub>)</h2>
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

        """<h2>8. Inverse Round-Trip Accuracy</h2>
        <p>A key advantage of this VDF over BPR: the flow-to-density inversion has a
        <b>closed-form solution</b> via the quadratic formula — no Newton solver needed.
        Left panel: q<sub>in</sub> vs q<sub>out</sub> after q → k(q) → q(k) round-trip
        (should fall exactly on the diagonal). Right panel: absolute error, which should
        be near machine epsilon (~10<sup>-10</sup>). This confirms the vectorized NumPy
        implementation is numerically exact.</p>""",

        """<h2>9. Speed–Density by Road Class</h2>
        <p>The bi-parabolic model is "parameter-light" — only v<sub>f</sub> (free-flow speed)
        and k<sub>j</sub> (jam density) are needed per link. k<sub>c</sub> = k<sub>j</sub>/3 is derived,
        not calibrated. This overlay shows how different road classes produce different
        curves from just those two inputs. Motorways have higher v<sub>f</sub> and k<sub>j</sub>
        (more lanes × higher jam density per lane), while residential streets have lower
        values of both. The shape is consistent across classes — only the scale changes.</p>""",

        f"""<h2>10. Extended Cost Chain for Flow-Based Assignment</h2>
        <p>The MFD inverse (§6) only exists for $q \\leq q_c$. For <b>flow-based
        assignment</b>, we need a monotone mapping from <em>any</em> demand level to
        a unique cost. The extended q → k mapping provides this:</p>

        <p><b>Uncongested</b> ($q \\leq q_c$): &emsp;
        $k(q) = k_c \\left(1 - \\sqrt{{1 - \\dfrac{{q}}{{q_c}}}}\\right)$
        &emsp; (exact MFD inverse, maps $[0, q_c] \\to [0, k_c]$)</p>

        <p><b>Oversaturated</b> ($q > q_c$): &emsp;
        $k(q) = k_c + (k_j - k_c)\\sqrt{{1 - \\dfrac{{q_c}}{{q}}}}$
        &emsp; (asymptotic extension, maps $(q_c, \\infty) \\to (k_c, k_j)$)</p>

        <p>The two pieces join at $q = q_c$ where both yield $k = k_c = {k_c:.0f}$.
        As $q \\to \\infty$, density approaches $k_j = {k_j:.0f}$ asymptotically —
        speed approaches zero but never reaches it, giving an unbounded, monotone
        cost function $t(q) = L / v(k(q))$.</p>

        <p><b>Left panel:</b> The piecewise q → k mapping. The blue uncongested branch
        is the exact MFD inverse; the red extension smoothly carries density toward
        $k_j$ for oversaturated demand. The dotted gray curve shows the MFD's
        congested-branch inverse for reference — it only exists for $q \\leq q_c$
        and maps to densities above $k_c$.</p>

        <p><b>Center panel:</b> Speed as a function of demand. The dotted gray curves show
        the classic backward-bending MFD parametric; the solid green line is the single-valued
        extended mapping — speed drops monotonically with demand.</p>

        <p><b>Right panel:</b> All candidate cost functions compared on a log scale.
        The <b>MFD cost chain</b> (green) is our physically-grounded approach. For
        comparison: <b>BPR(0.15,4)</b> — the standard calibrated function;
        <b>queue-delay</b> — adds $H/2 \\cdot (q/q_c - 1)$ delay above capacity;
        <b>steep-tail</b> (power {tail_power:.0f}) — a numerically simple power-law.
        The gray dotted curves show the backward-bending MFD parametric.
        Near capacity the MFD cost chain is steepest (best rerouting signal);
        all four surrogates are monotone and guarantee MSA convergence.</p>""",
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

        <p><b>Inverse MFD</b> ($0 \\leq q \\leq q_c$):</p>
        <p>Free-flow: &emsp; $k(q) = k_c \\left(1 - \\sqrt{{1 - \\dfrac{{2q}}{{v_f k_c}}}}\\right)$</p>
        <p>Congested: &emsp; $k(q) = k_c + (k_j - k_c)\\sqrt{{1 - \\dfrac{{2q}}{{v_f k_c}}}}$</p>

        <p><b>Static surrogate options</b> (single-valued demand-based costs):</p>
        <p>Queue-delay: &emsp;</p>
        <p>$$t^{{queue}}(q) =
        \\begin{{cases}}
        t_{{run}}(q), & q \\leq q_c \\\\
        t_c + \\dfrac{{H}}{{2}}\\left(\\dfrac{{q}}{{q_c}} - 1\\right), & q &gt; q_c
        \\end{{cases}}$$</p>
        <p>Steep tail: &emsp;</p>
        <p>$$t^{{tail}}(q) =
        \\begin{{cases}}
        t_{{run}}(q), & q \\leq q_c \\\\
        t_c \\left(\\dfrac{{q}}{{q_c}}\\right)^{{{tail_power:.0f}}}, & q &gt; q_c
        \\end{{cases}}$$</p>

        <p><b>Wardrop relative gap:</b> &emsp;
        $\\text{{gap}} = \\dfrac{{\\sum_a V_a \\cdot t_a}}{{\\sum_{{rs}} d_{{rs}} \\cdot \\pi_{{rs}}}} - 1$
        &emsp; where $t_a = L_a / v_a$ is link travel time, $\\pi_{{rs}}$ is shortest-path cost.</p>

        <p><b>Extended demand → density mapping</b> (§10, for flow-based assignment):</p>
        <p>$$k(q) =
        \\begin{{cases}}
        k_c \\left(1 - \\sqrt{{1 - \\dfrac{{q}}{{q_c}}}}\\right), & q \\leq q_c \\\\[6pt]
        k_c + (k_j - k_c)\\sqrt{{1 - \\dfrac{{q_c}}{{q}}}}, & q > q_c
        \\end{{cases}}$$</p>
        <p>Cost chain: $q \\xrightarrow{{k(q)}} k \\xrightarrow{{v(k)}} v \\xrightarrow{{L/v}} t$
        &emsp; — monotone for all $q \\geq 0$, guaranteeing MSA convergence.</p>
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
            # Skip fixedrange/dragmode overrides for zoomable figures
            if fig.layout.dragmode not in (None, False):
                fig.update_layout(
                    margin=dict(l=60, r=40, t=50, b=50),
                    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                )
            else:
                fig.update_layout(
                    margin=dict(l=60, r=40, t=50, b=50),
                    xaxis=dict(fixedrange=True),
                    yaxis=dict(fixedrange=True),
                    dragmode=False,
                    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
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
    path: str = "plots/assignment_report.html",
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
            network_state.density_vpkm * network_state.speed_kmh,
            network_state.density_vpkm,
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
    output_dir: str = "plots",
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
        network_state.density_vpkm * network_state.speed_kmh,
        network_state.density_vpkm,
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
        width=600, height=600,
        xaxis=dict(range=[0, max_val]),
        yaxis=dict(range=[0, max_val], scaleanchor="x", scaleratio=1),
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
    output_path: str = "plots/validation.html",
    max_iter: int = 50,
    detail_scale: float = 0.15,
    sweep_scales: Sequence[float] | None = None,
    vc_scales: Sequence[float] | None = None,
    methods: Sequence[str] | None = None,
    intro_html: str = "",
) -> Path:
    """Generate a standard validation report for any TNTP network.

    Produces an interactive HTML report with:

    0. Congestion map (links colored by k/k_j, hover for details)
    1. Demand scaling sweep (oversaturation, speed, TSTT)
    2. FW vs MSA convergence comparison
    3. Speed–density and flow–density MFD scatter
    4. Flow and travel time correlation vs BPR reference
    5. Per-link state table

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
        Deprecated — V/C scatter removed. Kept for API compatibility.
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


    # --- 1. Demand scaling sweep (skip if ≤1 scale) ---
    if len(sweep_scales) > 1:
        logger.debug(
            "[%s] Running demand sweep (%d scales)...",
            network_name, len(sweep_scales),
        )
        _add_sweep_section(
            figs, descriptions, base, meta, copy_fn, run_fn,
            sweep_scales, total_demand, n_links, max_iter,
        )

    # --- 2. FW vs MSA convergence ---
    run_methods = list(methods) if methods else ["fw", "msa"]
    logger.debug(
        "[%s] Running convergence comparison (%s, %d iters)...",
        network_name, "+".join(run_methods), max_iter,
    )
    result_fw = _add_convergence_section(
        figs, descriptions, base, meta, copy_fn, prepare_fn, run_fn,
        tmp_path, detail_scale, total_demand, max_iter, run_methods,
    )

    # --- 3. MFD scatter ---
    state = result_fw.network_state
    logger.debug("[%s] Building MFD scatter plots...", network_name)
    _add_mfd_section(figs, descriptions, state, detail_scale)

    # --- Insert congestion map + table at position 0 (before sweep) ---
    logger.debug("[%s] Building congestion map...", network_name)
    map_figs: list[go.Figure | None] = []
    map_descs: list[str] = []
    _add_congestion_map_section(
        map_figs, map_descs, network_name, node_coords, state,
        link_attrs, meta, detail_scale,
    )

    logger.debug("[%s] Building link state table...", network_name)
    _add_link_table_section(
        map_figs, map_descs, state, link_attrs, node_coords,
        detail_scale, total_demand,
    )
    figs[0:0] = map_figs
    descriptions[0:0] = map_descs

    # --- 4. Flow and TT correlation ---
    if ref:
        logger.debug("[%s] Building correlation plots...", network_name)
        _add_correlation_section(
            figs, descriptions, state, ref, link_attrs, detail_scale,
        )


    # Compose intro
    lane_counts = [a["n_lanes"] for a in link_attrs.values()]
    speed_set = sorted(set(round(a["ff_speed_kmh"]) for a in link_attrs.values()))
    if len(speed_set) <= 5:
        speed_desc = ", ".join(str(s) for s in speed_set)
    else:
        speed_desc = f"{speed_set[0]}&ndash;{speed_set[-1]} ({len(speed_set)} unique)"
    auto_intro = (
        f"<p>Validation of density-based traffic assignment on the "
        f"<b>{network_name}</b> benchmark "
        f"({len(node_coords)} nodes, {n_links} links, {n_zones} zones). "
        f"Total TNTP demand: {total_demand:,.0f} vph. "
        f"Lanes: {min(lane_counts)}&ndash;{max(lane_counts)}. "
        f"Freeflow speeds: {speed_desc} km/h. "
        f"VDF: bi-parabolic MFD (k<sub>j</sub>=150 veh/km/lane). "
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
            hovertext=f"{u}→{v}: {lanes}L, {attrs['ff_speed_kmh']:.0f} km/h",
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


def _add_congestion_map_section(figs, descriptions, name, nodes, state,
                                 link_attrs, meta, detail_scale):
    """Network map with links colored by k/k_j density ratio."""
    fig = go.Figure()

    # Build state edge lookup
    edge_map = {}
    for i in range(state.n_edges):
        u, v = int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1])
        edge_map[(u, v)] = i

    # Group links by color bucket for efficient rendering
    color_buckets = {
        "#4CAF50": {"label": "k/kj < 0.33", "xs": [], "ys": []},
        "#FF9800": {"label": "0.33 ≤ k/kj < 0.66", "xs": [], "ys": []},
        "#F44336": {"label": "0.66 ≤ k/kj < 0.90", "xs": [], "ys": []},
        "#B71C1C": {"label": "k/kj ≥ 0.90", "xs": [], "ys": []},
        "#BDBDBD": {"label": "No demand", "xs": [], "ys": []},
    }
    mid_x, mid_y, mid_color, mid_hover = [], [], [], []

    for (u, v), attrs in link_attrs.items():
        if u not in nodes or v not in nodes:
            continue
        x0, y0 = nodes[u]
        x1, y1 = nodes[v]
        lanes = attrs["n_lanes"]

        idx = edge_map.get((u, v))
        if idx is not None:
            kj = state.jam_density[idx]
            k = state.density_vpkm[idx]
            k_ratio = k / kj if kj > 0 else 0
            vf = state.freeflow_kmh[idx]
            v_cong = state.speed_kmh[idx]
            flow = state.flow_vph[idx]

            if k_ratio < 0.33:
                color = "#4CAF50"
            elif k_ratio < 0.66:
                color = "#FF9800"
            elif k_ratio < 0.90:
                color = "#F44336"
            else:
                color = "#B71C1C"

            hover = (
                f"{u}→{v}<br>"
                f"Lanes: {lanes}<br>"
                f"Freeflow: {vf:.1f} km/h<br>"
                f"Speed: {v_cong:.1f} km/h<br>"
                f"Density: {k:.1f} veh/km (k/kj={k_ratio:.2f})<br>"
                f"Flow: {flow:.0f} veh/hr"
            )
        else:
            color = "#BDBDBD"
            hover = (
                f"{u}→{v}<br>"
                f"Lanes: {lanes}<br>"
                f"Freeflow: {attrs['ff_speed_kmh']:.0f} km/h<br>"
                f"(no demand routed)"
            )

        # Append line segment with None separator for batching
        bucket = color_buckets[color]
        bucket["xs"].extend([x0, x1, None])
        bucket["ys"].extend([y0, y1, None])

        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_color.append(color)
        mid_hover.append(hover)

    # One trace per color bucket instead of one per link
    for color, bucket in color_buckets.items():
        if not bucket["xs"]:
            continue
        fig.add_trace(go.Scatter(
            x=bucket["xs"], y=bucket["ys"], mode="lines",
            line=dict(color=color, width=1),
            hoverinfo="skip",
            showlegend=False,
            name=bucket["label"],
        ))

    # Single invisible midpoint marker trace for hover
    fig.add_trace(go.Scatter(
        x=mid_x, y=mid_y, mode="markers",
        marker=dict(size=8, color=mid_color, opacity=0),
        hovertext=mid_hover, hoverinfo="text",
        showlegend=False,
    ))

    # Node markers
    xs = [nodes[n][0] for n in sorted(nodes)]
    ys = [nodes[n][1] for n in sorted(nodes)]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            size=5,
            color="white",
            line=dict(width=1, color="#333"),
        ),
        hovertext=[f"Node {n}" for n in sorted(nodes)],
        hoverinfo="text",
        showlegend=False,
    ))

    # Zone centroids (trip origins/destinations)
    centroids = meta.get("zone_centroids", {})
    od = meta.get("od_matrix")
    if centroids and od is not None:
        orig_vol = od.sum(axis=1) * detail_scale
        dest_vol = od.sum(axis=0) * detail_scale
        zx, zy, zvol, zhover = [], [], [], []
        for z_id, (lon, lat) in sorted(centroids.items()):
            zi = z_id - 1
            ov = float(orig_vol[zi]) if zi < len(orig_vol) else 0
            dv = float(dest_vol[zi]) if zi < len(dest_vol) else 0
            zx.append(lon)
            zy.append(lat)
            zvol.append(ov + dv)
            zhover.append(
                f"Zone {z_id}<br>"
                f"Origins: {ov:,.0f} vph<br>"
                f"Destinations: {dv:,.0f} vph"
            )
        fig.add_trace(go.Scatter(
            x=zx, y=zy, mode="markers",
            marker=dict(
                size=8, color=zvol, opacity=0.8,
                colorscale="Viridis",
                colorbar=dict(title="OD Volume (vph)"),
                symbol="circle",
                line=dict(width=1, color="white"),
            ),
            hovertext=zhover, hoverinfo="text",
            name="Zone centroids",
            showlegend=False,
        ))

    fig.update_layout(
        title=f"{name} Congestion Map ({detail_scale:.0%} demand)",
        xaxis=dict(title="Longitude", scaleanchor="y"),
        yaxis=dict(title="Latitude"),
        template="plotly_white",
        height=500,
        dragmode="zoom",
    )
    figs.append(fig)

    n_unused = len(link_attrs) - len(edge_map)
    n_crit = int(np.sum(state.density_vpkm > state.jam_density / 3))
    n_jam = int(np.sum(state.density_vpkm >= state.jam_density * 0.9))
    descriptions.append(
        f"<h2>Congestion Map ({detail_scale:.0%} demand)</h2>"
        f"<p>Links colored by density ratio k/k<sub>j</sub>: "
        f'<span style="color:#4CAF50"><b>green</b></span> (&lt; 0.33), '
        f'<span style="color:#FF9800"><b>yellow</b></span> (0.33–0.66), '
        f'<span style="color:#F44336"><b>red</b></span> (0.66–0.90), '
        f'<span style="color:#B71C1C"><b>dark red</b></span> (&ge; 0.90), '
        f'<span style="color:#BDBDBD"><b>grey</b></span> (no demand). '
        f"{n_crit} links above k<sub>c</sub>, {n_jam} near jam, "
        f"{n_unused} unused. Hover over links for details.</p>"
    )


def _add_sweep_section(figs, descriptions, base, meta, copy_fn, run_fn,
                        scales, total_demand, n_links, max_iter):
    """Section 1: demand scaling sweep (oversat, speed, TSTT)."""
    sweep_demand = []
    sweep_oversat = []
    sweep_mean_speed = []
    sweep_tstt = []

    for scale in scales:
        logger.debug("  Sweep scale %.0f%%...", scale * 100)
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
            f"{d:,.0f} vph → {o}/{n_links} links oversat"
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
                              detail_scale, total_demand, max_iter, methods):
    """Section 2: convergence plot. Returns first method's result."""
    colors = {"fw": "#D32F2F", "msa": "#1565C0"}
    labels = {"fw": "Frank-Wolfe", "msa": "MSA (1/n)"}
    dashes = {"fw": "solid", "msa": "dash"}
    results = {}

    for method in methods:
        logger.debug("  Running %s at %.0f%% demand...", method.upper(), detail_scale * 100)
        if method == methods[0]:
            run_base = copy_fn(base, tmp_path / f"{method}_detail")
            results[method] = run_fn(run_base, meta, max_iter, method, detail_scale)
        else:
            base_m, meta_m = prepare_fn(tmp_path / method)
            results[method] = run_fn(base_m, meta_m, max_iter, method, detail_scale)

    fig = go.Figure()
    for method in methods:
        iters = [r.iteration for r in results[method].iteration_log]
        gaps = [r.relative_gap for r in results[method].iteration_log]
        fig.add_trace(go.Scatter(
            x=iters, y=[abs(g) if g != 0 else None for g in gaps],
            mode="lines+markers", name=labels.get(method, method),
            line=dict(color=colors.get(method, "#666"), width=2.5 if method == methods[0] else 1.5,
                      dash=dashes.get(method, "solid")),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title=f"Wardrop Gap: {' vs '.join(labels.get(m, m) for m in methods)} ({detail_scale:.0%} Demand)",
        xaxis_title="Iteration", yaxis_title="|Relative Gap|",
        yaxis_type="log",
        template="plotly_white",
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True, dtick=1, tickformat=".0e"),
    )
    figs.append(fig)

    parts = []
    for method in methods:
        final = results[method].iteration_log[-1]
        parts.append(f"{labels.get(method, method)} gap: {final.relative_gap:.4f}")
    descriptions.append(
        f"<h2>Convergence at {detail_scale:.0%} Demand</h2>"
        f"<p>{', '.join(parts)} ({max_iter} iterations). "
        f"Demand: {total_demand * detail_scale:,.0f} vph "
        f"({detail_scale:.0%} of TNTP).</p>"
    )

    return results[methods[0]]


def _add_link_table_section(figs, descriptions, state, link_attrs, nodes,
                             detail_scale, total_demand):
    """Section 3: per-link state table."""
    rows = []
    midpoints = {}
    for i in range(state.n_edges):
        key = (int(state.edge_ids[i, 0]), int(state.edge_ids[i, 1]))
        la = link_attrs.get(key, {})
        dist_m = la.get("distance_m", 0)
        vf = state.freeflow_kmh[i]
        v = state.speed_kmh[i]
        flow = state.flow_vph[i]
        k = state.density_vpkm[i]
        ln = int(state.n_lanes[i])
        kj = state.jam_density[i]
        qc = vf * kj / 6
        tt = dist_m / 1000 / v * 60 if v > 0 else float("inf")
        ff_tt = dist_m / 1000 / vf * 60 if vf > 0 else float("inf")
        vc = flow / qc if qc > 0 else 0
        rows.append((key, ln, dist_m, vf, v, flow, k, kj, qc, ff_tt, tt, vc))
        u, vn = key
        if u in nodes and vn in nodes:
            mx = (nodes[u][0] + nodes[vn][0]) / 2
            my = (nodes[u][1] + nodes[vn][1]) / 2
            midpoints[f"{u}→{vn}"] = (mx, my)

    rows.sort(key=lambda r: r[0])

    def _vc_color(vc):
        if vc > 0.85:
            return "color:#D32F2F;font-weight:bold;"
        if vc > 0.6:
            return "color:#FF6F00;"
        return ""

    table_html = (
        '<div style="margin-bottom:8px;">'
        '<input type="text" id="linkFilter" placeholder="Filter links (e.g. 397)" '
        'oninput="filterTable()" '
        'style="padding:4px 8px;font-size:13px;border:1px solid #ccc;border-radius:3px;width:200px;">'
        '</div>'
        '<div style="max-height:500px;overflow-y:auto;border:1px solid #ddd;">'
        '<table id="linkTable" style="border-collapse:collapse; width:100%; font-size:13px; '
        'font-family:monospace;">\n'
        '<thead style="position:sticky;top:0;background:#fff;z-index:1;">'
        '<tr style="border-bottom:2px solid #333;cursor:pointer;" '
        'onclick="sortTable(event)">'
        '<th data-col="0" style="text-align:left;padding:6px;">Link</th>'
        '<th data-col="1" style="padding:6px;">Lanes</th>'
        '<th data-col="2" style="padding:6px;">Dist (m)</th>'
        '<th data-col="3" style="padding:6px;">v<sub>f</sub></th>'
        '<th data-col="4" style="padding:6px;">q<sub>c</sub></th>'
        '<th data-col="5" style="padding:6px;border-left:2px solid #ccc;">Flow</th>'
        '<th data-col="6" style="padding:6px;">k</th>'
        '<th data-col="7" style="padding:6px;">k/k<sub>j</sub></th>'
        '<th data-col="8" style="padding:6px;">V/C</th>'
        '<th data-col="9" style="padding:6px;">Speed</th>'
        '<th data-col="10" style="padding:6px;">FF TT</th>'
        '<th data-col="11" style="padding:6px;">TT</th>'
        '<th data-col="12" style="padding:6px;">TT/FF</th>'
        '</tr></thead>\n<tbody>\n'
    )

    for key, ln, dist_m, vf, v, flow, k, kj, qc, ff_tt, tt, vc in rows:
        ratio = tt / ff_tt if ff_tt > 0 and tt < float("inf") else float("inf")
        vc_style = _vc_color(vc)
        k_kj = k / kj if kj > 0 else 0
        k_style = ("color:#D32F2F;font-weight:bold;" if k_kj > 0.9
                    else "color:#FF6F00;" if k_kj > 0.5 else "")
        tt_str = f"{tt:.2f}" if tt < float("inf") else "&infin;"
        ratio_str = f"{ratio:.2f}" if ratio < float("inf") else "&infin;"
        link_id = f"{key[0]}→{key[1]}"
        table_html += (
            f'<tr style="border-bottom:1px solid #eee;cursor:pointer;" '
            f'data-link="{link_id}" onclick="highlightLink(this)">'
            f'<td style="padding:4px 6px;">{key[0]}→{key[1]}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{ln}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{dist_m:.0f}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{vf:.0f}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{qc:.0f}</td>'
            f'<td style="text-align:right;padding:4px 6px;border-left:2px solid #ccc;">{flow:.0f}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{k:.1f}</td>'
            f'<td style="text-align:right;padding:4px 6px;{k_style}">{k_kj:.2f}</td>'
            f'<td style="text-align:right;padding:4px 6px;{vc_style}">{vc:.2f}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{v:.1f}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{ff_tt:.2f}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{tt_str}</td>'
            f'<td style="text-align:right;padding:4px 6px;">{ratio_str}</td>'
            f'</tr>\n'
        )

    table_html += '</tbody></table></div>'

    # Embed midpoint coordinates for map highlighting
    import json
    mp_json = json.dumps({k: list(v) for k, v in midpoints.items()})

    table_html += f'''
<script>
var sortDir = {{}};
var linkMidpoints = {mp_json};
var _prevHighlight = null;

function sortTable(e) {{
    var th = e.target.closest('th');
    if (!th) return;
    var col = parseInt(th.dataset.col);
    var table = document.getElementById('linkTable');
    var tbody = table.tBodies[0];
    var rows = Array.from(tbody.rows);
    sortDir[col] = !(sortDir[col] || false);
    var asc = sortDir[col] ? 1 : -1;
    rows.sort(function(a, b) {{
        var at = a.cells[col].textContent.trim();
        var bt = b.cells[col].textContent.trim();
        var an = parseFloat(at), bn = parseFloat(bt);
        if (!isNaN(an) && !isNaN(bn)) return (an - bn) * asc;
        return at.localeCompare(bt) * asc;
    }});
    rows.forEach(function(r) {{ tbody.appendChild(r); }});
}}

function filterTable() {{
    var val = document.getElementById('linkFilter').value.toLowerCase();
    var rows = document.getElementById('linkTable').tBodies[0].rows;
    for (var i = 0; i < rows.length; i++) {{
        var txt = rows[i].cells[0].textContent.toLowerCase();
        rows[i].style.display = txt.indexOf(val) >= 0 ? '' : 'none';
    }}
}}

function highlightLink(tr) {{
    if (_prevHighlight) _prevHighlight.style.background = '';
    tr.style.background = '#FFF9C4';
    _prevHighlight = tr;

    var linkId = tr.dataset.link;
    var pt = linkMidpoints[linkId];
    if (!pt) return;

    var mapDiv = document.getElementById('fig-0');
    if (!mapDiv) return;

    // Annotation only — no zoom
    Plotly.relayout(mapDiv, {{
        annotations: [{{
            x: pt[0], y: pt[1],
            xref: 'x', yref: 'y',
            text: '<b>' + linkId + '</b>',
            showarrow: true,
            arrowhead: 2, arrowsize: 1.5, arrowcolor: '#000',
            font: {{ size: 12, color: '#000' }},
            bgcolor: '#FFF9C4',
            bordercolor: '#333',
            borderwidth: 1,
            borderpad: 3,
        }}],
    }});

    mapDiv.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
}}
</script>'''

    all_vc = [r[11] for r in rows]
    all_speeds = [r[4] for r in rows]
    finite_ratios = [r[10] / r[9] for r in rows if r[9] > 0 and r[10] < float("inf")]
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


def _add_mfd_section(figs, descriptions, state, detail_scale):
    """Section: network-wide MFD from per-link state.

    Plots speed–density and flow–density scatter for all links,
    overlaid with theoretical VDF curves for each lane class present
    in the network.
    """
    from osrm.assignment.vdf import BiParabolicVDF
    vdf = BiParabolicVDF(min_speed_kmh=1.08)

    k = state.density_vpkm
    v = state.speed_kmh
    q = state.density_vpkm * state.speed_kmh  # physical throughput
    vf = state.freeflow_kmh
    kj = state.jam_density
    n_lanes = state.n_lanes

    # Per-lane normalization for comparable visualization
    k_per_lane = k / np.maximum(n_lanes, 1)
    q_per_lane = q / np.maximum(n_lanes, 1)
    kj_per_lane = kj / np.maximum(n_lanes, 1)

    # --- Single VDF curve normalized per lane ---
    med_vf = float(np.median(vf))
    med_kj_lane = float(np.median(kj_per_lane))
    k_pts = np.linspace(0, med_kj_lane, 200)
    vf_arr = np.full_like(k_pts, med_vf)
    kj_arr = np.full_like(k_pts, med_kj_lane)
    v_pts = vdf.density_to_speed(k_pts, vf_arr, kj_arr)
    q_pts = vdf.density_to_flow(k_pts, vf_arr, kj_arr)
    curve_label = f"VDF (v_f={med_vf:.0f}, k_j={med_kj_lane:.0f}/lane)"

    # --- Combined Speed–Density and Flow–Density (per lane) ---
    hover_labels = [
        f"{int(state.edge_ids[i,0])}→{int(state.edge_ids[i,1])} "
        f"({int(n_lanes[i])}L)"
        for i in range(state.n_edges)
    ]
    fig_mfd = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Speed–Density", "Flow–Density", "Unserved Demand"],
    )
    # Left: speed–density scatter
    fig_mfd.add_trace(go.Scatter(
        x=k_per_lane.tolist(), y=v.tolist(), mode="markers",
        marker=dict(size=4, color="#1565C0", opacity=0.5),
        hovertext=hover_labels, hoverinfo="text",
        showlegend=False,
    ), row=1, col=1)
    fig_mfd.add_trace(go.Scatter(
        x=k_pts.tolist(), y=v_pts.tolist(), mode="lines",
        line=dict(color="#999", dash="dash", width=1),
        name=curve_label, showlegend=True,
    ), row=1, col=1)
    # Right: flow–density scatter
    fig_mfd.add_trace(go.Scatter(
        x=k_per_lane.tolist(), y=q_per_lane.tolist(), mode="markers",
        marker=dict(size=4, color="#D32F2F", opacity=0.5),
        hovertext=hover_labels, hoverinfo="text",
        showlegend=False,
    ), row=1, col=2)
    fig_mfd.add_trace(go.Scatter(
        x=k_pts.tolist(), y=q_pts.tolist(), mode="lines",
        line=dict(color="#999", dash="dash", width=1),
        name=curve_label, showlegend=False,
    ), row=1, col=2)
    # Right: unserved demand (queue buildup rate) vs density
    q_unserved = state.unserved_demand
    q_unserved_per_lane = q_unserved / np.maximum(n_lanes, 1)
    fig_mfd.add_trace(go.Scatter(
        x=k_per_lane.tolist(), y=q_unserved_per_lane.tolist(), mode="markers",
        marker=dict(size=4, color="#FF6F00", opacity=0.5),
        hovertext=hover_labels, hoverinfo="text",
        showlegend=False,
    ), row=1, col=3)
    fig_mfd.update_xaxes(title_text="Density per lane (veh/km/lane)", rangemode="tozero")
    fig_mfd.update_yaxes(rangemode="tozero")
    fig_mfd.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig_mfd.update_yaxes(title_text="Flow per lane (veh/hr/lane)", row=1, col=2)
    fig_mfd.update_yaxes(title_text="Unserved (veh/hr/lane)", row=1, col=3)
    fig_mfd.update_layout(
        title=f"Macroscopic Fundamental Diagram ({detail_scale:.0%} demand)",
        template="plotly_white",
        height=450, width=1400,
    )
    figs.append(fig_mfd)
    n_queued = int(np.sum(q_unserved > 0))
    total_unserved = float(np.sum(q_unserved))
    descriptions.append(
        "<h2>Macroscopic Fundamental Diagram (MFD)</h2>"
        "<p>Each point is one link, normalized to per-lane density so links "
        "with different lane counts are comparable. Dashed curve: theoretical "
        f"VDF using median v<sub>f</sub>={med_vf:.0f} km/h, "
        f"k<sub>j</sub>={med_kj_lane:.0f} veh/km/lane. "
        "Left: speed–density. Centre: flow–density (inverted-U fundamental diagram). "
        f"Right: unserved demand = demand − physical throughput. "
        f"{n_queued}/{state.n_edges} links have unserved demand "
        f"(total {total_unserved:,.0f} veh/hr).</p>"
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
        link_labels.append(f"{key[0]}→{key[1]}")

    # Compute correlations
    flow_rho = 0.0
    if len(mfd_flows) >= 3:
        flow_rho, _ = spearmanr(bpr_flows, mfd_flows)
    flow_max = max(max(bpr_flows, default=1), max(mfd_flows, default=1)) * 1.1

    finite_mask = [t < float("inf") for t in mfd_tt]
    tt_bpr_f = [b for b, m in zip(bpr_tt, finite_mask) if m]
    tt_mfd_f = [t for t, m in zip(mfd_tt, finite_mask) if m]
    tt_labels_f = [l for l, m in zip(link_labels, finite_mask) if m]
    n_gridlocked = sum(1 for m in finite_mask if not m)
    tt_rho = 0.0
    if len(tt_mfd_f) >= 3:
        tt_rho, _ = spearmanr(tt_bpr_f, tt_mfd_f)
    tt_max = max(max(tt_bpr_f, default=1), max(tt_mfd_f, default=1)) * 1.1

    # Combined correlation figure
    fig_corr = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Flow Correlation (ρ={flow_rho:.3f})",
            f"Travel Time Correlation (ρ={tt_rho:.3f})",
        ],
    )
    # Left: flow scatter
    fig_corr.add_trace(go.Scatter(
        x=bpr_flows, y=mfd_flows, mode="markers",
        marker=dict(size=6, color="#1565C0", opacity=0.7),
        hovertext=link_labels, hoverinfo="text",
        showlegend=False,
    ), row=1, col=1)
    fig_corr.add_shape(
        type="line", x0=0, x1=flow_max, y0=0, y1=flow_max,
        line=dict(color="#999", dash="dash", width=1),
        row=1, col=1,
    )
    # Right: TT scatter
    fig_corr.add_trace(go.Scatter(
        x=tt_bpr_f, y=tt_mfd_f, mode="markers",
        marker=dict(size=6, color="#D32F2F", opacity=0.7),
        hovertext=tt_labels_f, hoverinfo="text",
        showlegend=False,
    ), row=1, col=2)
    if tt_bpr_f and tt_mfd_f:
        fig_corr.add_shape(
            type="line", x0=0, x1=tt_max, y0=0, y1=tt_max,
            line=dict(color="#999", dash="dash", width=1),
            row=1, col=2,
        )
    fig_corr.update_xaxes(title_text="BPR Flow (vph, 100%)", range=[0, flow_max], row=1, col=1)
    fig_corr.update_yaxes(title_text=f"MFD Flow (vph, {detail_scale:.0%})", range=[0, flow_max], row=1, col=1)
    fig_corr.update_xaxes(title_text="BPR Travel Time (min)", range=[0, tt_max] if tt_bpr_f else None, row=1, col=2)
    fig_corr.update_yaxes(title_text=f"MFD Travel Time (min, {detail_scale:.0%})", range=[0, tt_max] if tt_bpr_f else None, row=1, col=2)
    fig_corr.update_layout(
        title=f"Correlation: MFD ({detail_scale:.0%} demand) vs BPR (100%)",
        template="plotly_white",
        height=450, width=1000,
    )
    figs.append(fig_corr)
    gridlock_note = (
        f" ({n_gridlocked} gridlocked link{'s' if n_gridlocked != 1 else ''} excluded.)"
        if n_gridlocked > 0 else ""
    )
    descriptions.append(
        "<h2>Correlation: MFD vs BPR</h2>"
        "<p>Scatter of per-link flow (left) and travel time (right): our density-based "
        f"MFD assignment ({detail_scale:.0%} demand, {state.n_edges} links) vs published "
        "BPR equilibrium (100% demand). Dashed line = y&thinsp;=&thinsp;x. "
        f"Flow &rho; = <b>{flow_rho:.3f}</b> ({len(mfd_flows)} matched links). "
        f"Travel time &rho; = <b>{tt_rho:.3f}</b> ({len(tt_mfd_f)} links).{gridlock_note}</p>"
    )


def _add_vc_section(figs, descriptions, base, meta, copy_fn, run_fn,
                     link_attrs, scales, max_iter, tmp_path):
    """Section 5: V/C scatter across demand levels."""
    fig = go.Figure()
    colors = ["#4CAF50", "#1565C0", "#FF6F00", "#D32F2F",
              "#9C27B0", "#00BCD4", "#795548", "#607D8B"]
    max_flow = 1

    for sc, col in zip(scales, colors):
        logger.debug("  V/C scale %.0f%%...", sc * 100)
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
            hover_list.append(f"{k[0]}→{k[1]}: V/C={vc_i:.2f}")
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
