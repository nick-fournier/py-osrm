"""Shared traffic assignment validation helpers.

These helpers provide network prep, metadata, and plotting primitives for
MSA-based validation reports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import plotly.graph_objects as go

from osrm.assignment import (
    AssignmentConfig,
    DemandTrip,
    DensitySmoothingConfig,
    AssignmentSolver,
)
from osrm.assignment.plots import (
    _add_congestion_map_section,
    _add_correlation_section,
    _add_link_table_section,
    _add_mfd_section,
    _write_combined_report,
)


@dataclass
class ValidationCase:
    """Materialized assignment validation run."""

    base_path: str
    meta: dict
    trips: list[DemandTrip]
    result: object
    total_demand: float


def run_validation_case(
    *,
    base_path: str,
    meta: dict,
    copy_fn,
    trip_builder: Callable[[dict, float], list[DemandTrip]],
    run_dir: Path,
    demand_scale: float = 1.0,
    state_patch_factory: Callable[[dict], Callable] | None = None,
    max_rounds: int = 0,
    gap_threshold: float = 0.001,
    method: str = "msa",
):
    """Run one shared assignment validation case from scenario metadata."""
    logger = logging.getLogger(__name__)

    logger.info("Building trips (demand_scale=%.2f)...", demand_scale)
    t0 = time.monotonic()
    trips = trip_builder(meta, demand_scale)
    logger.info("Built %d OD pairs in %.1fs", len(trips), time.monotonic() - t0)

    run_base = copy_fn(base_path, run_dir)
    config = AssignmentConfig(
        method=method,
        max_iterations=max_rounds if max_rounds > 0 else 20,
        convergence_gap=gap_threshold,
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(Path(run_base).parent),
    )
    solver = AssignmentSolver(run_base, config)
    state_patch = state_patch_factory(meta) if state_patch_factory else None
    result = solver.assign_matrix(trips, state_patch=state_patch)
    return ValidationCase(
        base_path=run_base,
        meta=meta,
        trips=trips,
        result=result,
        total_demand=sum(t.volume for t in trips),
    )


def validation_final_gap(result: object) -> float | None:
    """Return the final convergence gap, if available."""
    gap = getattr(result, "final_gap", None)
    if gap is not None:
        return float(gap)
    iteration_log = getattr(result, "iteration_log", []) or []
    if iteration_log:
        return iteration_log[-1].relative_gap
    return None


def validation_final_tstt(
    result: object,
    *,
    min_speed_kmh: float = 1.08,
) -> float:
    """Return final TSTT from iteration log."""
    iteration_log = getattr(result, "iteration_log", []) or []
    if iteration_log:
        return float(iteration_log[-1].tstt)
    return 0.0


def iteration_series(result: object) -> dict:
    """Extract plotting arrays from an AssignmentResult's iteration_log."""
    iteration_log = getattr(result, "iteration_log", []) or []
    if not iteration_log:
        return {
            "network_tstt": [], "state_change_norm": [], "speeds": [],
            "max_k": [], "median_k": [], "route_times": [],
            "customize_times": [], "engine_times": [], "gaps": [],
        }
    return {
        "network_tstt": [r.tstt for r in iteration_log],
        "state_change_norm": [r.state_change_norm for r in iteration_log],
        "speeds": [r.mean_speed_kmh for r in iteration_log],
        "max_k": [getattr(r, "max_k_over_kj", 0.0) for r in iteration_log],
        "median_k": [getattr(r, "median_k_over_kj", 0.0) for r in iteration_log],
        "route_times": [r.route_time_s for r in iteration_log],
        "customize_times": [r.customize_time_s for r in iteration_log],
        "engine_times": [r.engine_time_s for r in iteration_log],
        "gaps": [r.relative_gap for r in iteration_log],
    }


def _add_batch_sections(
    figs: list[go.Figure | None],
    descriptions: list[str],
    case: ValidationCase,
    *,
    detail_scale: float,
) -> None:
    """Add convergence, state evolution, and runtime plots."""
    result = case.result
    iteration_log = getattr(result, "iteration_log", []) or []
    if not iteration_log:
        return

    labels = [f"Iter {r.iteration}" for r in iteration_log]
    tstt_vals = [r.tstt for r in iteration_log]
    gap_vals = [r.relative_gap for r in iteration_log]
    delta_vals = [r.state_change_norm for r in iteration_log]
    speeds = [r.mean_speed_kmh for r in iteration_log]
    max_k = [getattr(r, "max_k_over_kj", 0.0) for r in iteration_log]
    median_k = [getattr(r, "median_k_over_kj", 0.0) for r in iteration_log]

    # --- Convergence: TSTT + Gap + Δq ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=tstt_vals, name="TSTT",
        mode="lines+markers",
        line=dict(color="#D32F2F", width=2.5),
        marker=dict(size=7),
    ))
    if any(d != 0.0 for d in delta_vals):
        fig.add_trace(go.Scatter(
            x=labels, y=delta_vals, name="\u0394q norm",
            mode="lines+markers",
            line=dict(color="#8E24AA", width=2.5, dash="dot"),
            marker=dict(size=7), yaxis="y3",
        ))
    fig.add_trace(go.Scatter(
        x=labels, y=gap_vals, name="Gap", yaxis="y2",
        mode="lines+markers",
        line=dict(color="#1976D2", width=2.5),
        marker=dict(size=7), connectgaps=False,
    ))
    fig.update_layout(
        title="Convergence",
        xaxis_title="Iteration",
        yaxis=dict(title="Network TSTT (veh-seconds)"),
        yaxis2=dict(
            title="Gap", overlaying="y", side="right",
            type="log" if any(g is not None for g in gap_vals) else "linear",
        ),
        yaxis3=dict(
            title="\u0394q norm", anchor="free",
            overlaying="y", side="right", position=0.92,
        ),
        template="plotly_white",
    )
    figs.append(fig)
    final_gap = validation_final_gap(result)
    gap_str = f"{final_gap:.6f}" if final_gap is not None else "n/a"
    final_tstt = validation_final_tstt(result)
    descriptions.append(
        "<h2>Convergence</h2>"
        "<p>MSA iterations blend all-or-nothing auxiliary loadings "
        "with diminishing step sizes (\u03b1 = 1/(m+1)) to converge toward "
        "user equilibrium. "
        f"Final TSTT = {final_tstt:,.0f} veh-seconds; final gap = {gap_str}.</p>"
    )

    # --- State Evolution: speed + k/kj ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=speeds, mode="lines+markers",
        line=dict(color="#2E7D32", width=2.5),
        marker=dict(size=7), name="Mean speed",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=max_k, mode="lines+markers",
        line=dict(color="#FF9800", width=2.0),
        marker=dict(size=6), name="Max k/kj", yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=median_k, mode="lines+markers",
        line=dict(color="#FF9800", width=1.5, dash="dash"),
        marker=dict(size=5), name="Median k/kj (loaded)", yaxis="y2",
    ))
    fig.update_layout(
        title="State Evolution",
        xaxis_title="Iteration",
        yaxis=dict(title="Mean speed (km/h)"),
        yaxis2=dict(title="k / k_jam", overlaying="y", side="right"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    figs.append(fig)
    n_iter = len(iteration_log)
    descriptions.append(
        f"<h2>State Evolution</h2>"
        f"<p>Final loaded demand is {case.total_demand:,.0f} vph. "
        f"MSA ran {n_iter} iteration(s)."
        f" Final gap={gap_str}.</p>"
    )

    # --- Runtime: per-iteration timings ---
    rt_route = [r.route_time_s for r in iteration_log]
    rt_cust = [r.customize_time_s for r in iteration_log]
    rt_engine = [r.engine_time_s for r in iteration_log]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=rt_route, mode="lines+markers",
        line=dict(color="#1565C0", width=2.0),
        marker=dict(size=5), name="Route time",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=rt_cust, mode="lines+markers",
        line=dict(color="#8E24AA", width=2.0),
        marker=dict(size=5), name="Customize time",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=rt_engine, mode="lines+markers",
        line=dict(color="#6D4C41", width=2.0),
        marker=dict(size=5), name="Engine reload time",
    ))
    fig.update_layout(
        title="Runtime",
        xaxis_title="Iteration",
        yaxis_title="Time (s)",
        template="plotly_white",
    )
    figs.append(fig)
    descriptions.append(
        "<h2>Runtime</h2>"
        "<p>Per-iteration routing, customize, and engine reload timings.</p>"
    )


def generate_validation_report(
    *,
    network_name: str,
    prepare_fn,
    copy_fn,
    trip_builder: Callable[[dict, float], list[DemandTrip]],
    tmp_path: str | Path,
    output_path: str,
    detail_scale: float,
    state_patch_factory: Callable[[dict], Callable] | None = None,
    intro_html: str = "",
    max_rounds: int = 0,
    gap_threshold: float = 0.001,
    method: str = "msa",
) -> Path:
    """Generate a shared assignment validation report for one scenario."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    base, meta = prepare_fn(tmp_path)
    case = run_validation_case(
        base_path=base,
        meta=meta,
        copy_fn=copy_fn,
        trip_builder=trip_builder,
        run_dir=tmp_path / "hc_detail",
        demand_scale=detail_scale,
        state_patch_factory=state_patch_factory,
        max_rounds=max_rounds,
        gap_threshold=gap_threshold,
        method=method,
    )

    state = case.result.network_state
    if state is None:
        raise RuntimeError(f"{network_name} assignment produced no network state")

    node_coords = meta["nodes"]
    link_attrs = meta["link_attrs"]
    ref = meta.get("ref_flows", {})

    figs: list[go.Figure | None] = []
    descriptions: list[str] = []

    _add_congestion_map_section(
        figs,
        descriptions,
        network_name,
        node_coords,
        state,
        link_attrs,
        meta,
        detail_scale,
    )
    _add_link_table_section(
        figs,
        descriptions,
        state,
        link_attrs,
        node_coords,
        detail_scale,
        case.total_demand,
    )
    _add_batch_sections(figs, descriptions, case, detail_scale=detail_scale)
    _add_mfd_section(figs, descriptions, state, detail_scale)

    if ref:
        _add_correlation_section(figs, descriptions, state, ref, link_attrs, detail_scale)

    lane_counts = [a["n_lanes"] for a in link_attrs.values()]
    speed_set = sorted(set(round(a["ff_speed_kmh"]) for a in link_attrs.values()))
    if len(speed_set) <= 5:
        speed_desc = ", ".join(str(s) for s in speed_set)
    else:
        speed_desc = f"{speed_set[0]}&ndash;{speed_set[-1]} ({len(speed_set)} unique)"

    converge_msg = ""
    if max_rounds > 0:
        converge_msg = (
            f" {method.upper()} convergence runs "
            f"for up to {max_rounds} round(s)."
        )

    intro = (
        f"<p>Validation of the <b>{method.upper()} assignment</b> on "
        f"<b>{network_name}</b>. Final loaded demand: {case.total_demand:,.0f} vph "
        f"({detail_scale:.0%} of the scenario demand basis). "
        f"Trips: {len(case.trips):,} OD pairs. "
        f"Lanes: {min(lane_counts)}&ndash;{max(lane_counts)}. "
        f"Freeflow speeds: {speed_desc} km/h. "
        f"Total runtime: {case.result.total_time_s:.2f}s.{converge_msg}</p>"
    )
    if intro_html:
        intro += intro_html

    _write_combined_report(
        title=f"{network_name} Validation",
        intro=intro,
        figures=figs,
        descriptions=descriptions,
        path=Path(output_path),
    )
    return Path(output_path)


def build_validation_report_sections(
    *,
    network_name: str,
    case: ValidationCase,
    detail_scale: float,
) -> tuple[list[go.Figure | None], list[str]]:
    """Build the shared report sections used by assignment validations."""
    state = case.result.network_state
    if state is None:
        raise RuntimeError(f"{network_name} assignment produced no network state")

    meta = case.meta
    node_coords = meta["nodes"]
    link_attrs = meta["link_attrs"]
    ref = meta.get("ref_flows", {})

    figs: list[go.Figure | None] = []
    descriptions: list[str] = []

    _add_congestion_map_section(
        figs,
        descriptions,
        network_name,
        node_coords,
        state,
        link_attrs,
        meta,
        detail_scale,
    )
    _add_link_table_section(
        figs,
        descriptions,
        state,
        link_attrs,
        node_coords,
        detail_scale,
        case.total_demand,
    )
    _add_batch_sections(figs, descriptions, case, detail_scale=detail_scale)
    _add_mfd_section(figs, descriptions, state, detail_scale)

    if ref:
        _add_correlation_section(figs, descriptions, state, ref, link_attrs, detail_scale)

    return figs, descriptions
