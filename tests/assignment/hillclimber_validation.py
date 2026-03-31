"""Shared hill-climber validation helpers.

These helpers intentionally reuse the same network prep, metadata, and plotting
primitives as the matrix validation path so the two validation surfaces do not
drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import plotly.graph_objects as go

from osrm.assignment import (
    AssignmentConfig,
    DemandTrip,
    DensitySmoothingConfig,
    MatrixFreeHillClimber,
)
from osrm.assignment.plots import (
    _add_congestion_map_section,
    _add_correlation_section,
    _add_link_table_section,
    _add_mfd_section,
    _write_combined_report,
)


@dataclass
class HillClimberValidationCase:
    """Materialized hill-climber validation run."""

    base_path: str
    meta: dict
    trips: list[DemandTrip]
    sliced_trips: list[DemandTrip]
    result: object
    total_demand: float


def slice_trips_by_departure(
    trips: Sequence[DemandTrip],
    *,
    n_slices: int,
    bin_width_s: float,
) -> list[DemandTrip]:
    """Split each OD trip evenly across fixed departure-time slices."""
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if bin_width_s <= 0:
        raise ValueError("bin_width_s must be positive")

    sliced: list[DemandTrip] = []
    for trip in trips:
        per_slice = trip.volume / n_slices
        if per_slice <= 0:
            continue
        for slice_idx in range(n_slices):
            sliced.append(DemandTrip(
                origin=trip.origin,
                destination=trip.destination,
                volume=per_slice,
                departure_time_s=slice_idx * bin_width_s,
            ))
    return sliced


def run_hillclimber_case(
    *,
    base_path: str,
    meta: dict,
    copy_fn,
    trip_builder: Callable[[dict, float], list[DemandTrip]],
    run_dir: Path,
    demand_scale: float = 1.0,
    n_slices: int = 4,
    bin_width_s: float = 3600.0,
    max_batch_size: int | None = None,
    state_patch_factory: Callable[[dict], Callable] | None = None,
    max_epochs: int = 0,
    gap_threshold: float = 0.01,
):
    """Run one shared hill-climber validation case from scenario metadata."""
    trips = trip_builder(meta, demand_scale)
    sliced_trips = slice_trips_by_departure(
        trips,
        n_slices=n_slices,
        bin_width_s=bin_width_s,
    )
    run_base = copy_fn(base_path, run_dir)
    config = AssignmentConfig(
        bin_width_s=bin_width_s,
        smoothing=DensitySmoothingConfig(method="none"),
        speed_csv_dir=str(Path(run_base).parent),
    )
    solver = MatrixFreeHillClimber(
        run_base,
        config,
        default_batch_size=max_batch_size or len(sliced_trips),
    )
    state_patch = state_patch_factory(meta) if state_patch_factory else None
    result = solver.run_stream(
        sliced_trips,
        max_batch_size=max_batch_size,
        state_patch=state_patch,
        max_epochs=max_epochs,
        gap_threshold=gap_threshold,
    )
    return HillClimberValidationCase(
        base_path=run_base,
        meta=meta,
        trips=trips,
        sliced_trips=sliced_trips,
        result=result,
        total_demand=sum(t.volume for t in trips),
    )


def _add_batch_sections(
    figs: list[go.Figure | None],
    descriptions: list[str],
    case: HillClimberValidationCase,
    *,
    detail_scale: float,
) -> None:
    """Add hill-climber slice evolution plots."""
    result = case.result
    slice_labels = [f"E0:S{b.batch_index}" for b in result.batch_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=slice_labels,
        y=[b.n_trips for b in result.batch_results],
        name="Trips",
        marker_color="#1976D2",
    ))
    fig.add_trace(go.Scatter(
        x=slice_labels,
        y=[b.batch_tstt for b in result.batch_results],
        name="Slice TSTT",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#D32F2F", width=2.5),
    ))
    fig.update_layout(
        title="Slice Timeline",
        xaxis_title="Slice",
        yaxis=dict(title="Trips"),
        yaxis2=dict(
            title="Slice TSTT (veh-seconds)",
            overlaying="y",
            side="right",
        ),
        template="plotly_white",
    )
    figs.append(fig)
    descriptions.append(
        "<h2>Slice Timeline</h2>"
        "<p>Trips are loaded sequentially by departure slice on a shared mutable "
        "network state. Bars show trips per slice; the line shows per-slice total "
        "system travel time.</p>"
    )

    # --- State Evolution: unified E0:S0 … E0:SN → E1:S0 … E1:SN timeline ---
    fig = go.Figure()

    # Build unified x-axis labels and y-values
    greedy_labels = [f"E0:S{b.batch_index}" for b in result.batch_results]
    greedy_speeds = [b.mean_speed_kmh for b in result.batch_results]
    greedy_k = [b.max_k_over_kj for b in result.batch_results]

    epoch_results = getattr(result, "epoch_results", []) or []
    reroute_labels: list[str] = []
    reroute_speeds: list[float] = []
    reroute_k: list[float] = []
    gap_annotations: list[tuple[int, float]] = []  # (x_index, gap_value)

    for er in epoch_results:
        for snap in er.slice_snapshots:
            reroute_labels.append(f"E{er.epoch}:S{snap.slice_index}")
            reroute_speeds.append(snap.mean_speed_kmh)
            reroute_k.append(snap.max_k_over_kj)
        if er.gap is not None:
            gap_annotations.append((
                len(greedy_labels) + len(reroute_labels) - 1,
                er.gap,
            ))

    all_labels = greedy_labels + reroute_labels
    all_speeds = greedy_speeds + reroute_speeds
    all_k = greedy_k + reroute_k

    # Mean speed trace (continuous)
    fig.add_trace(go.Scatter(
        x=all_labels,
        y=all_speeds,
        mode="lines+markers",
        line=dict(color="#2E7D32", width=2.5),
        marker=dict(size=7),
        name="Mean speed",
    ))
    # Max k/kj trace (continuous)
    fig.add_trace(go.Scatter(
        x=all_labels,
        y=all_k,
        mode="lines+markers",
        line=dict(color="#FF9800", width=2.0),
        marker=dict(size=6),
        name="Max k/kj",
        yaxis="y2",
    ))

    # Vertical separator between greedy and reroute phases
    if epoch_results:
        fig.add_vline(
            x=len(greedy_labels) - 0.5,
            line_dash="dot", line_color="#666", line_width=1.5,
            annotation_text="⟵ greedy | reroute ⟶",
            annotation_position="top",
        )

    # Gap annotations at epoch boundaries
    for x_idx, gap_val in gap_annotations:
        fig.add_annotation(
            x=all_labels[x_idx], y=all_speeds[x_idx],
            text=f"gap={gap_val:.4f}",
            showarrow=True, arrowhead=2, arrowcolor="#999",
            font=dict(size=9, color="#666"),
            yshift=15,
        )

    fig.update_layout(
        title="State Evolution",
        xaxis_title="Slice (Epoch:Slice)",
        yaxis=dict(title="Mean speed (km/h)"),
        yaxis2=dict(title="Max k/kj", overlaying="y", side="right"),
        template="plotly_white",
    )
    figs.append(fig)
    epoch_note = ""
    if epoch_results:
        n_reroute = sum(er.slices_rerouted for er in epoch_results)
        final_gap = epoch_results[-1].gap
        gap_str = f" Converged after {len(epoch_results)} epoch(s), final gap={final_gap:.6f}." if final_gap is not None else ""
        epoch_note = (
            f" After greedy loading (E0), {len(epoch_results)} reroute epoch(s) "
            f"re-processed all {result.batch_results[-1].batch_index + 1} slices "
            f"({n_reroute} total reroutes). Gap computed once per epoch.{gap_str}"
        )
    descriptions.append(
        f"<h2>State Evolution</h2>"
        f"<p>Final loaded demand is {case.total_demand:,.0f} "
        "vph, distributed deterministically across departure slices."
        f"{epoch_note}</p>"
    )

    # --- Runtime: unified per-slice timing across all epochs ---
    epoch_results = getattr(result, "epoch_results", []) or []

    # Build unified x-labels and timing arrays
    rt_labels = [f"E0:S{b.batch_index}" for b in result.batch_results]
    rt_route = [b.route_time_s for b in result.batch_results]
    rt_cust = [b.customize_time_s for b in result.batch_results]
    rt_engine = [b.engine_time_s for b in result.batch_results]

    for er in epoch_results:
        for snap in er.slice_snapshots:
            rt_labels.append(f"E{er.epoch}:S{snap.slice_index}")
            rt_route.append(snap.route_time_s)
            rt_cust.append(snap.customize_time_s)
            rt_engine.append(snap.engine_time_s)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rt_labels, y=rt_route,
        mode="lines+markers",
        line=dict(color="#1565C0", width=2.0),
        marker=dict(size=5),
        name="Route time",
    ))
    fig.add_trace(go.Scatter(
        x=rt_labels, y=rt_cust,
        mode="lines+markers",
        line=dict(color="#8E24AA", width=2.0),
        marker=dict(size=5),
        name="Customize time",
    ))
    fig.add_trace(go.Scatter(
        x=rt_labels, y=rt_engine,
        mode="lines+markers",
        line=dict(color="#6D4C41", width=2.0),
        marker=dict(size=5),
        name="Engine reload time",
    ))

    # Vertical lines at epoch boundaries with gap on hover
    n_greedy = len(result.batch_results)
    for er in epoch_results:
        x_end = n_greedy + er.epoch * len(er.slice_snapshots) - 0.5
        gap_str = f"gap={er.gap:.6f}" if er.gap is not None else "gap=n/a"
        fig.add_vline(
            x=x_end, line_dash="dot", line_color="#E65100", line_width=1.5,
            annotation_text=f"E{er.epoch} {gap_str}",
            annotation_position="top",
            annotation_font_size=9,
            annotation_font_color="#E65100",
        )
    # Separator between greedy and reroute
    if epoch_results:
        fig.add_vline(
            x=n_greedy - 0.5,
            line_dash="dot", line_color="#666", line_width=1.5,
        )

    fig.update_layout(
        title="Slice Runtime",
        xaxis_title="Slice (Epoch:Slice)",
        yaxis_title="Time (s)",
        template="plotly_white",
    )
    figs.append(fig)
    descriptions.append(
        "<h2>Slice Runtime</h2>"
        "<p>Per-slice routing, customize, and engine reload timings across all epochs. "
        "Vertical lines mark epoch boundaries with the Wardrop gap at that point.</p>"
    )


def generate_hillclimber_validation_report(
    *,
    network_name: str,
    prepare_fn,
    copy_fn,
    trip_builder: Callable[[dict, float], list[DemandTrip]],
    tmp_path: str | Path,
    output_path: str,
    detail_scale: float,
    n_slices: int = 4,
    bin_width_s: float = 3600.0,
    max_batch_size: int | None = None,
    state_patch_factory: Callable[[dict], Callable] | None = None,
    intro_html: str = "",
) -> Path:
    """Generate a shared hill-climber validation report for one scenario."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    base, meta = prepare_fn(tmp_path)
    case = run_hillclimber_case(
        base_path=base,
        meta=meta,
        copy_fn=copy_fn,
        trip_builder=trip_builder,
        run_dir=tmp_path / "hc_detail",
        demand_scale=detail_scale,
        n_slices=n_slices,
        bin_width_s=bin_width_s,
        max_batch_size=max_batch_size,
        state_patch_factory=state_patch_factory,
    )

    state = case.result.network_state
    if state is None:
        raise RuntimeError(f"{network_name} hill-climber produced no network state")

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
    intro = (
        f"<p>Validation of the <b>matrix-free hill-climber MVP</b> on "
        f"<b>{network_name}</b>. Final loaded demand: {case.total_demand:,.0f} vph "
        f"({detail_scale:.0%} of the scenario demand basis), distributed across "
        f"{n_slices} departure slices of {bin_width_s / 3600:.1f} hour(s) each. "
        f"Trips: {len(case.trips):,} OD movements materialized as "
        f"{len(case.sliced_trips):,} departure-sliced loads. "
        f"Lanes: {min(lane_counts)}&ndash;{max(lane_counts)}. "
        f"Freeflow speeds: {speed_desc} km/h. "
        f"Total runtime: {case.result.total_time_s:.2f}s.</p>"
    )
    if intro_html:
        intro += intro_html

    _write_combined_report(
        title=f"{network_name} Hill-Climber Validation",
        intro=intro,
        figures=figs,
        descriptions=descriptions,
        path=Path(output_path),
    )
    return Path(output_path)


def build_hillclimber_report_sections(
    *,
    network_name: str,
    case: HillClimberValidationCase,
    detail_scale: float,
) -> tuple[list[go.Figure | None], list[str]]:
    """Build the shared report sections used by hill-climber validations."""
    state = case.result.network_state
    if state is None:
        raise RuntimeError(f"{network_name} hill-climber produced no network state")

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
