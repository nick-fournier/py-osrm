"""Shared hill-climber validation helpers.

These helpers intentionally reuse the same network prep, metadata, and plotting
primitives as the matrix validation path so the two validation surfaces do not
drift apart.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
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

_DEFAULT_ASSUMED_SPEED_KMH = 30.0


@dataclass
class HillClimberValidationCase:
    """Materialized hill-climber validation run."""

    base_path: str
    meta: dict
    trips: list[DemandTrip]
    sliced_trips: list[DemandTrip]
    result: object
    total_demand: float
    load_steps: int


def _median_link_speed_from_meta(meta: dict) -> float | None:
    """Extract median freeflow speed from meta['link_attrs'] if available."""
    link_attrs = meta.get("link_attrs")
    if not link_attrs:
        return None
    speeds = [a["ff_speed_kmh"] for a in link_attrs.values() if "ff_speed_kmh" in a]
    if not speeds:
        return None
    return float(np.median(speeds))


def compute_volume_threshold(
    *,
    median_speed_kmh: float,
    jam_density_per_lane: float = 150.0,
    n_lanes: int = 1,
    n_slices: int,
    bin_width_hr: float = 1.0,
) -> float:
    """Per-card volume threshold: each card contributes ≤ kj/n_slices density.

    ``threshold = (kj / n_slices) * speed * bin_width``

    A single OD pair routed at this volume adds at most ``kj / n_slices``
    density to every link on its path, preventing any one card from
    oversaturating the network.
    """
    kj = jam_density_per_lane * n_lanes
    return (kj / n_slices) * median_speed_kmh * bin_width_hr


def slice_trips_by_departure(
    trips: Sequence[DemandTrip],
    *,
    n_slices: int,
    bin_width_s: float,
    volume_threshold: float,
    seed: int = 42,
) -> list[DemandTrip]:
    """Replicate-shuffle-deal OD trips into departure-time slices.

    Dual-axis slicing separates spatial OD sampling from volumetric
    demand splitting:

    1. **Replicate** – OD pairs with ``volume > volume_threshold`` are
       split into ``ceil(volume / volume_threshold)`` cards, each
       carrying an equal share of the original volume.  Low-volume
       pairs produce one card at full volume.
    2. **Shuffle** – The full deck is randomly permuted (seeded for
       reproducibility) so each slice gets a spatially diverse mix.
    3. **Deal** – The shuffled deck is divided into *n_slices* contiguous
       blocks, each assigned a successive departure time.

    On large spread networks (e.g. Chicago Regional) most OD pairs have
    sub-threshold volume, so the deck ≈ n_od_pairs and spatial sampling
    dominates.  On small concentrated networks (e.g. Braess) the single
    high-volume pair is replicated into many cards and volumetric
    splitting dominates.
    """
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if bin_width_s <= 0:
        raise ValueError("bin_width_s must be positive")
    if volume_threshold <= 0:
        raise ValueError("volume_threshold must be positive")

    positive_trips = [t for t in trips if t.volume > 0]
    n_ods = len(positive_trips)
    if n_ods == 0:
        return []

    # Per-OD minimum cards so every slice gets at least one card
    spatial_floor = max(1, math.ceil(n_slices / n_ods))

    # Step 1: Replicate high-volume pairs
    origins: list = []
    destinations: list = []
    volumes: list[float] = []
    for trip in positive_trips:
        n_cards = max(spatial_floor, math.ceil(trip.volume / volume_threshold))
        per_card = trip.volume / n_cards
        for _ in range(n_cards):
            origins.append(trip.origin)
            destinations.append(trip.destination)
            volumes.append(per_card)

    n = len(origins)

    # Step 2: Shuffle
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    # Step 3: Deal into slices
    sliced: list[DemandTrip] = []
    for rank, idx in enumerate(perm):
        slice_idx = min(rank * n_slices // n, n_slices - 1)
        sliced.append(DemandTrip(
            origin=origins[idx],
            destination=destinations[idx],
            volume=volumes[idx],
            departure_time_s=slice_idx * bin_width_s,
        ))
    return sliced


def _sampled_load_steps(sample_rate: float) -> int:
    """Derive greedy load-step count from sample_rate.

    Sampled refinement treats the greedy initializer as repeated equal-volume
    load steps, where each step size matches the refinement sample share.
    That means sample_rate must be the reciprocal of an integer step count:
    100% -> 1 step, 50% -> 2, 25% -> 4, 10% -> 10, etc.
    """
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError("sample_rate must be in (0, 1]")
    load_steps = max(1, int(round(1.0 / sample_rate)))
    if not math.isclose(sample_rate * load_steps, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "sampled mode requires sample_rate to be the reciprocal of an integer "
            "load-step count (for example 1.0, 0.5, 0.25, 0.2, 0.1, 0.05)"
        )
    return load_steps


def run_hillclimber_case(
    *,
    base_path: str,
    meta: dict,
    copy_fn,
    trip_builder: Callable[[dict, float], list[DemandTrip]],
    run_dir: Path,
    demand_scale: float = 1.0,
    bin_width_s: float = 3600.0,
    max_batch_size: int | None = None,
    state_patch_factory: Callable[[dict], Callable] | None = None,
    sample_rate: float,
    max_rounds: int = 0,
    gap_threshold: float = 0.01,
    assumed_speed_kmh: float | None = None,
):
    """Run one shared hill-climber validation case from scenario metadata."""
    logger = logging.getLogger(__name__)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    load_steps = _sampled_load_steps(sample_rate)

    logger.info("Building trips (demand_scale=%.2f)...", demand_scale)
    t0 = time.monotonic()
    trips = trip_builder(meta, demand_scale)
    logger.info("Built %d OD pairs in %.1fs", len(trips), time.monotonic() - t0)

    # Determine median network speed for volume threshold
    if assumed_speed_kmh is not None:
        median_speed = assumed_speed_kmh
    else:
        median_speed = _median_link_speed_from_meta(meta)
        if median_speed is None:
            median_speed = _DEFAULT_ASSUMED_SPEED_KMH
            logger.info(
                "No link_attrs in meta; using default assumed speed %.0f km/h",
                median_speed,
            )

    vol_threshold = compute_volume_threshold(
        median_speed_kmh=median_speed,
        n_slices=load_steps,
        bin_width_hr=bin_width_s / 3600.0,
    )
    n_above = sum(1 for t in trips if t.volume > vol_threshold)
    logger.info(
        "Volume threshold %.0f veh/hr (median speed %.1f km/h, %d slices); "
        "%d/%d OD pairs will be replicated",
        vol_threshold, median_speed, load_steps, n_above, len(trips),
    )

    logger.info("Replicate-shuffle-deal into %d load slices...", load_steps)
    t0 = time.monotonic()
    sliced_trips = slice_trips_by_departure(
        trips,
        n_slices=load_steps,
        bin_width_s=bin_width_s,
        volume_threshold=vol_threshold,
    )
    logger.info(
        "Dealt %d cards from %d OD pairs into %d slices in %.1fs",
        len(sliced_trips), len(trips), load_steps, time.monotonic() - t0,
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
        sample_rate=sample_rate,
        max_rounds=max_rounds,
        gap_threshold=gap_threshold,
    )
    return HillClimberValidationCase(
        base_path=run_base,
        meta=meta,
        trips=trips,
        sliced_trips=sliced_trips,
        result=result,
        total_demand=sum(t.volume for t in trips),
        load_steps=load_steps,
    )


def hillclimber_final_gap(result: object) -> float | None:
    """Return the final sampled-refinement gap, if available."""
    refinement_results = getattr(result, "refinement_results", []) or []
    if refinement_results:
        return refinement_results[-1].sampled_gap

    od_ledger = getattr(result, "od_ledger", None)
    if od_ledger is not None and len(od_ledger) > 0:
        numerator = 0.0
        denominator = 0.0
        for entry in od_ledger:
            if entry.current_gap is None:
                continue
            numerator += float(entry.total_volume) * float(entry.current_gap)
            denominator += float(entry.total_volume)
        if denominator > 0.0:
            return numerator / denominator

    return None


def hillclimber_final_tstt(
    result: object,
    *,
    min_speed_kmh: float = 1.08,
) -> float:
    """Estimate final TSTT from the active sampled path-set state."""
    refinement_results = getattr(result, "refinement_results", []) or []
    od_ledger = getattr(result, "od_ledger", None)
    state = getattr(result, "network_state", None)
    if refinement_results and od_ledger is not None and len(od_ledger) > 0 and state is not None:
        total_tstt = 0.0
        for entry in od_ledger:
            for route in entry.routes:
                route_cost_s = 0.0
                for edge_idx in route.edge_indices:
                    if edge_idx < 0 or edge_idx >= state.n_edges:
                        continue
                    speed_kmh = max(float(state.speed_kmh[edge_idx]), min_speed_kmh)
                    route_cost_s += float(state.length_m[edge_idx]) / (speed_kmh / 3.6)
                total_tstt += (
                    float(entry.total_volume)
                    * float(route.volume_fraction)
                    * route_cost_s
                )
        return float(total_tstt)

    batch_results = getattr(result, "batch_results", []) or []
    if batch_results:
        return float(batch_results[-1].network_tstt)
    return 0.0


def _load_step_label(batch_index: int) -> str:
    return f"Load {batch_index + 1}"


def _refinement_step_label(round_index: int) -> str:
    return f"Refine {round_index}"


def _refinement_series(result: object) -> dict:
    """Return plotting arrays for sampled path-set refinement, if present."""
    refinement_results = getattr(result, "refinement_results", []) or []
    if refinement_results:
        return {
            "kind": "refinement",
            "labels": [
                _refinement_step_label(round_result.round_index)
                for round_result in refinement_results
            ],
            "network_tstt": [round_result.network_tstt for round_result in refinement_results],
            "speeds": [round_result.mean_speed_kmh for round_result in refinement_results],
            "k": [round_result.max_k_over_kj for round_result in refinement_results],
            "route_times": [round_result.route_time_s for round_result in refinement_results],
            "customize_times": [round_result.customize_time_s for round_result in refinement_results],
            "engine_times": [round_result.engine_time_s for round_result in refinement_results],
            "gaps": [round_result.sampled_gap for round_result in refinement_results],
        }
    return {
        "kind": None,
        "labels": [],
        "network_tstt": [],
        "speeds": [],
        "k": [],
        "route_times": [],
        "customize_times": [],
        "engine_times": [],
        "gaps": [],
    }


def _add_batch_sections(
    figs: list[go.Figure | None],
    descriptions: list[str],
    case: HillClimberValidationCase,
    *,
    detail_scale: float,
) -> None:
    """Add hill-climber load-step evolution plots."""
    result = case.result
    greedy_labels = [_load_step_label(b.batch_index) for b in result.batch_results]
    greedy_tstt = [b.network_tstt for b in result.batch_results]
    refinement = _refinement_series(result)

    convergence_labels = greedy_labels + refinement["labels"]
    convergence_tstt = list(greedy_tstt)
    convergence_tstt.extend(refinement["network_tstt"])

    gap_values = [None] * len(greedy_labels) + refinement["gaps"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=convergence_labels,
        y=convergence_tstt,
        name="TSTT",
        mode="lines+markers",
        line=dict(color="#D32F2F", width=2.5),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=convergence_labels,
        y=gap_values,
        name="Gap",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#1976D2", width=2.5),
        marker=dict(size=7),
        connectgaps=False,
    ))
    if refinement["kind"] is not None:
        fig.add_vline(
            x=len(greedy_labels) - 0.5,
            line_dash="dot",
            line_color="#666",
            line_width=1.5,
        )
    fig.update_layout(
        title="Convergence",
        xaxis_title="Load / refinement step",
        yaxis=dict(title="Network TSTT (veh-seconds)"),
        yaxis2=dict(
            title="Gap",
            overlaying="y",
            side="right",
            type="log" if any(gap is not None for gap in gap_values) else "linear",
        ),
        template="plotly_white",
    )
    figs.append(fig)
    final_gap = hillclimber_final_gap(result)
    gap_str = f"{final_gap:.6f}" if final_gap is not None else "n/a"
    final_tstt = hillclimber_final_tstt(result)
    descriptions.append(
        "<h2>Convergence</h2>"
        "<p>Network TSTT is shown across greedy loading and any post-greedy refinement, "
        "with the corresponding gap on the secondary axis. This replaces the old "
        "slice-timeline view so convergence quality is explicit and comparable from start to finish. "
        f"Final TSTT = {final_tstt:,.0f} veh-seconds; final gap = {gap_str}.</p>"
    )

    # --- State Evolution: unified load / refinement timeline ---
    fig = go.Figure()

    # Build unified x-axis labels and y-values
    greedy_speeds = [b.mean_speed_kmh for b in result.batch_results]
    greedy_k = [b.max_k_over_kj for b in result.batch_results]
    all_labels = greedy_labels + refinement["labels"]
    all_speeds = greedy_speeds + refinement["speeds"]
    all_k = greedy_k + refinement["k"]

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

    # Vertical separator between greedy and post-greedy refinement phases
    if refinement["kind"] is not None:
        fig.add_vline(
            x=len(greedy_labels) - 0.5,
            line_dash="dot", line_color="#666", line_width=1.5,
        )

    # Gap annotations at refinement boundaries / rounds
    for ref_idx, gap_val in enumerate(refinement["gaps"]):
        if gap_val is None:
            continue
        x_idx = len(greedy_labels) + ref_idx
        fig.add_annotation(
            x=all_labels[x_idx], y=all_speeds[x_idx],
            text=f"gap={gap_val:.4f}",
            showarrow=True, arrowhead=2, arrowcolor="#999",
            font=dict(size=9, color="#666"),
            yshift=15,
        )

    fig.update_layout(
        title="State Evolution",
        xaxis_title="Load / refinement step",
        yaxis=dict(title="Mean speed (km/h)"),
        yaxis2=dict(title="Max k/kj", overlaying="y", side="right"),
        template="plotly_white",
    )
    figs.append(fig)
    refinement_note = ""
    if refinement["kind"] == "refinement":
        refinement_results = getattr(result, "refinement_results", []) or []
        accepted_updates = sum(round_result.accepted_updates for round_result in refinement_results)
        final_gap = hillclimber_final_gap(result)
        gap_str = (
            f" Final sampled gap={final_gap:.6f}."
            if final_gap is not None else ""
        )
        refinement_note = (
            f" After greedy loading, {len(refinement_results)} sampled refinement round(s) "
            f"rebalanced OD path sets with {accepted_updates} accepted OD updates."
            f"{gap_str}"
        )
    descriptions.append(
        f"<h2>State Evolution</h2>"
        f"<p>Final loaded demand is {case.total_demand:,.0f} "
        f"vph. Greedy loading ran across {case.load_steps} load step(s)."
        f"{refinement_note}</p>"
    )

    # --- Runtime: unified per-step timing ---
    rt_labels = [_load_step_label(b.batch_index) for b in result.batch_results]
    rt_route = [b.route_time_s for b in result.batch_results]
    rt_cust = [b.customize_time_s for b in result.batch_results]
    rt_engine = [b.engine_time_s for b in result.batch_results]
    rt_labels.extend(refinement["labels"])
    rt_route.extend(refinement["route_times"])
    rt_cust.extend(refinement["customize_times"])
    rt_engine.extend(refinement["engine_times"])

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

    # Vertical separator between greedy and refinement
    n_greedy = len(result.batch_results)
    if refinement["kind"] is not None:
        fig.add_vline(
            x=n_greedy - 0.5,
            line_dash="dot", line_color="#666", line_width=1.5,
        )

    fig.update_layout(
        title="Runtime",
        xaxis_title="Load / refinement step",
        yaxis_title="Time (s)",
        template="plotly_white",
    )
    figs.append(fig)
    runtime_note = (
        "Per-step routing, customize, and engine reload timings across greedy loading "
        "and sampled path-set refinement rounds."
        if refinement["kind"] == "refinement" else
        "Per-step routing, customize, and engine reload timings across greedy loading."
    )
    descriptions.append(
        "<h2>Runtime</h2>"
        f"<p>{runtime_note}</p>"
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
    bin_width_s: float = 3600.0,
    max_batch_size: int | None = None,
    state_patch_factory: Callable[[dict], Callable] | None = None,
    intro_html: str = "",
    sample_rate: float,
    max_rounds: int = 0,
    gap_threshold: float = 0.01,
    assumed_speed_kmh: float | None = None,
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
        bin_width_s=bin_width_s,
        max_batch_size=max_batch_size,
        state_patch_factory=state_patch_factory,
        sample_rate=sample_rate,
        max_rounds=max_rounds,
        gap_threshold=gap_threshold,
        assumed_speed_kmh=assumed_speed_kmh,
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
    sampled_mode = max_rounds > 0
    refinement_intro = ""
    if sampled_mode:
        refinement_intro = (
            f" Greedy loading uses {case.load_steps} load step(s) derived from the "
            f"{sample_rate:.0%} sample rate. Sampled path-set rebalancing then runs "
            f"for up to {max_rounds} round(s)."
        )
    else:
        refinement_intro = (
            f" Greedy loading uses {case.load_steps} load step(s) derived from the "
            f"{sample_rate:.0%} sample rate."
        )

    load_distribution = (
        f"loaded via replicate-shuffle-deal into {case.load_steps} greedy load step(s). "
        f"Trips: {len(case.trips):,} OD pairs dealt into "
        f"{len(case.sliced_trips):,} cards. "
    )

    intro = (
        f"<p>Validation of the <b>matrix-free hill-climber MVP</b> on "
        f"<b>{network_name}</b>. Final loaded demand: {case.total_demand:,.0f} vph "
        f"({detail_scale:.0%} of the scenario demand basis), {load_distribution}"
        f"Lanes: {min(lane_counts)}&ndash;{max(lane_counts)}. "
        f"Freeflow speeds: {speed_desc} km/h. "
        f"Total runtime: {case.result.total_time_s:.2f}s.{refinement_intro}</p>"
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
