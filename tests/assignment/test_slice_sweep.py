"""Cross-network slice sweep: convergence sensitivity to N slices.

Runs the hill-climber at [2, 4, 8, 16, 32] slices on Sioux Falls, Anaheim,
and Chicago Sketch with reroute epochs (max_epochs=10, gap_threshold=0.001).
Produces a standalone HTML report comparing epochs-to-converge, final gap,
and total runtime as a function of slice count and network size.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from osrm.assignment.plots import _write_combined_report
from .hillclimber_validation import run_hillclimber_case


# ── Network descriptors ──────────────────────────────────────────────────────

@dataclass
class NetworkSpec:
    """Everything needed to run HC on a network."""
    name: str
    color: str
    prepare_fn: object
    copy_fn: object
    trip_builder: object
    detail_scale: float
    state_patch_factory: object | None = None
    n_nodes: int = 0
    n_links: int = 0


def _build_specs() -> list[NetworkSpec]:
    """Lazily import and return the three network specs."""
    from .test_sioux_falls import (
        _prepare_sf_network,
        _copy_clean_osrm as sf_copy,
        _build_hillclimber_trips as sf_trips,
    )
    from osrm.assignment.osm_synthesis import patch_sioux_falls_lanes

    from .test_anaheim import (
        _prepare_anaheim_network,
        _copy_clean_osrm as ana_copy,
        _build_hillclimber_trips as ana_trips,
    )
    from osrm.assignment.osm_synthesis import patch_lanes

    from .test_chicago_sketch import (
        _prepare_chicago_network,
        _copy_clean_osrm as chi_copy,
        _build_hillclimber_trips as chi_trips,
    )

    return [
        NetworkSpec(
            name="Sioux Falls",
            color="#1565C0",
            prepare_fn=_prepare_sf_network,
            copy_fn=sf_copy,
            trip_builder=sf_trips,
            detail_scale=0.15,
            state_patch_factory=lambda meta: lambda state: patch_sioux_falls_lanes(state, meta),
        ),
        NetworkSpec(
            name="Anaheim",
            color="#2E7D32",
            prepare_fn=_prepare_anaheim_network,
            copy_fn=ana_copy,
            trip_builder=ana_trips,
            detail_scale=1.00,
            state_patch_factory=lambda meta: lambda state: patch_lanes(state, meta),
        ),
        NetworkSpec(
            name="Chicago Sketch",
            color="#D32F2F",
            prepare_fn=_prepare_chicago_network,
            copy_fn=chi_copy,
            trip_builder=chi_trips,
            detail_scale=1.00,
            state_patch_factory=lambda meta: lambda state: patch_lanes(state, meta),
        ),
    ]


# ── Sweep data ────────────────────────────────────────────────────────────────

SWEEP_SLICES = [2, 4, 8, 16, 32]
MAX_EPOCHS = 10
GAP_THRESHOLD = 0.001


@dataclass
class SweepPoint:
    n_slices: int
    n_epochs: int
    final_gap: float | None
    total_time_s: float
    greedy_tstt: float
    final_tstt: float


@dataclass
class NetworkSweepResult:
    spec: NetworkSpec
    points: list[SweepPoint] = field(default_factory=list)


def _run_sweep(spec: NetworkSpec, tmp_root: Path) -> NetworkSweepResult:
    """Run the slice sweep for one network."""
    nsr = NetworkSweepResult(spec=spec)
    base, meta = spec.prepare_fn(tmp_root / spec.name.lower().replace(" ", "_"))
    spec.n_nodes = len(meta["nodes"])
    spec.n_links = len(meta["link_attrs"])

    for ns in SWEEP_SLICES:
        t0 = time.perf_counter()
        case = run_hillclimber_case(
            base_path=base,
            meta=meta,
            copy_fn=spec.copy_fn,
            trip_builder=spec.trip_builder,
            run_dir=tmp_root / f"{spec.name.lower().replace(' ', '_')}_s{ns}",
            demand_scale=spec.detail_scale,
            n_slices=ns,
            max_epochs=MAX_EPOCHS,
            gap_threshold=GAP_THRESHOLD,
            state_patch_factory=spec.state_patch_factory,
        )
        elapsed = time.perf_counter() - t0
        result = case.result
        epochs = result.epoch_results or []
        n_epochs = len(epochs)
        final_gap = epochs[-1].gap if epochs else None

        greedy_tstt = sum(b.batch_tstt for b in result.batch_results)
        if epochs and epochs[-1].slice_snapshots:
            final_tstt = sum(s.tstt for s in epochs[-1].slice_snapshots)
        else:
            final_tstt = greedy_tstt

        nsr.points.append(SweepPoint(
            n_slices=ns,
            n_epochs=n_epochs,
            final_gap=final_gap,
            total_time_s=elapsed,
            greedy_tstt=greedy_tstt,
            final_tstt=final_tstt,
        ))
    return nsr


# ── Report generation ─────────────────────────────────────────────────────────

def _build_report_sections(
    results: list[NetworkSweepResult],
) -> tuple[list[go.Figure | None], list[str]]:
    """Build all figures for the cross-network sweep report."""
    figs: list[go.Figure | None] = []
    descriptions: list[str] = []

    # ── 1. Epochs to converge vs N slices ─────────────────────────────────
    fig_epochs = go.Figure()
    for nsr in results:
        spec = nsr.spec
        fig_epochs.add_trace(go.Scatter(
            x=[p.n_slices for p in nsr.points],
            y=[p.n_epochs for p in nsr.points],
            mode="lines+markers",
            line=dict(color=spec.color, width=2.5),
            marker=dict(size=8),
            name=f"{spec.name} ({spec.n_nodes}n/{spec.n_links}e)",
            hovertemplate=(
                f"{spec.name}<br>"
                "slices=%{x}<br>epochs=%{y}<extra></extra>"
            ),
        ))
    fig_epochs.update_layout(
        title="Reroute Epochs to Converge vs Slice Count",
        xaxis_title="Number of departure slices",
        yaxis_title="Epochs to converge",
        template="plotly_white",
        xaxis=dict(type="log", tickvals=SWEEP_SLICES,
                   ticktext=[str(s) for s in SWEEP_SLICES]),
        yaxis=dict(rangemode="tozero"),
    )
    figs.append(fig_epochs)
    descriptions.append(
        "<h2>Epochs to Converge</h2>"
        "<p>Number of reroute epochs before the Wardrop gap drops below "
        f"{GAP_THRESHOLD} (or max {MAX_EPOCHS} reached). "
        "More slices generally require fewer epochs because each slice "
        "carries less demand and causes smaller perturbations.</p>"
    )

    # ── 2. Final gap vs N slices ──────────────────────────────────────────
    fig_gap = go.Figure()
    for nsr in results:
        spec = nsr.spec
        gaps = [p.final_gap if p.final_gap is not None else 0.0 for p in nsr.points]
        fig_gap.add_trace(go.Scatter(
            x=[p.n_slices for p in nsr.points],
            y=gaps,
            mode="lines+markers",
            line=dict(color=spec.color, width=2.5),
            marker=dict(size=8),
            name=f"{spec.name}",
            hovertemplate=(
                f"{spec.name}<br>"
                "slices=%{x}<br>gap=%{y:.6f}<extra></extra>"
            ),
        ))
    fig_gap.add_hline(
        y=GAP_THRESHOLD, line_dash="dot", line_color="#999",
        annotation_text=f"threshold={GAP_THRESHOLD}",
        annotation_position="bottom right",
    )
    fig_gap.update_layout(
        title="Final Wardrop Gap vs Slice Count",
        xaxis_title="Number of departure slices",
        yaxis_title="Final gap (relative excess cost)",
        template="plotly_white",
        xaxis=dict(type="log", tickvals=SWEEP_SLICES,
                   ticktext=[str(s) for s in SWEEP_SLICES]),
        yaxis=dict(type="log"),
    )
    figs.append(fig_gap)
    descriptions.append(
        "<h2>Final Wardrop Gap</h2>"
        "<p>The gap after convergence (or at max epochs). Lower is better. "
        "The dotted line marks the convergence threshold. "
        "Points below the line converged; points above hit the epoch limit.</p>"
    )

    # ── 3. Runtime vs N slices ────────────────────────────────────────────
    fig_time = go.Figure()
    for nsr in results:
        spec = nsr.spec
        fig_time.add_trace(go.Scatter(
            x=[p.n_slices for p in nsr.points],
            y=[p.total_time_s for p in nsr.points],
            mode="lines+markers",
            line=dict(color=spec.color, width=2.5),
            marker=dict(size=8),
            name=f"{spec.name}",
            hovertemplate=(
                f"{spec.name}<br>"
                "slices=%{x}<br>time=%{y:.1f}s<extra></extra>"
            ),
        ))
    fig_time.update_layout(
        title="Total Runtime vs Slice Count",
        xaxis_title="Number of departure slices",
        yaxis_title="Total wall-clock time (s)",
        template="plotly_white",
        xaxis=dict(type="log", tickvals=SWEEP_SLICES,
                   ticktext=[str(s) for s in SWEEP_SLICES]),
        yaxis=dict(rangemode="tozero"),
    )
    figs.append(fig_time)
    descriptions.append(
        "<h2>Total Runtime</h2>"
        "<p>Wall-clock time including greedy loading and all reroute epochs. "
        "More slices = more OSRM route + customize calls, but potentially "
        "fewer epochs needed. The trade-off is network-size dependent.</p>"
    )

    # ── 4. TSTT improvement (greedy → rerouted) ──────────────────────────
    fig_tstt = go.Figure()
    for nsr in results:
        spec = nsr.spec
        pct_improve = [
            (p.greedy_tstt - p.final_tstt) / p.greedy_tstt * 100
            if p.greedy_tstt > 0 else 0.0
            for p in nsr.points
        ]
        fig_tstt.add_trace(go.Scatter(
            x=[p.n_slices for p in nsr.points],
            y=pct_improve,
            mode="lines+markers",
            line=dict(color=spec.color, width=2.5),
            marker=dict(size=8),
            name=f"{spec.name}",
            hovertemplate=(
                f"{spec.name}<br>"
                "slices=%{x}<br>TSTT improvement=%{y:.2f}%<extra></extra>"
            ),
        ))
    fig_tstt.update_layout(
        title="TSTT Improvement from Rerouting vs Slice Count",
        xaxis_title="Number of departure slices",
        yaxis_title="TSTT reduction (%)",
        template="plotly_white",
        xaxis=dict(type="log", tickvals=SWEEP_SLICES,
                   ticktext=[str(s) for s in SWEEP_SLICES]),
    )
    figs.append(fig_tstt)
    descriptions.append(
        "<h2>TSTT Improvement</h2>"
        "<p>Percentage reduction in total system travel time from reroute epochs "
        "relative to greedy-only loading. Larger improvements indicate the greedy "
        "solution was further from equilibrium and rerouting had more to gain.</p>"
    )

    # ── 5. Summary table ─────────────────────────────────────────────────
    header_cells = [
        "Network", "Nodes", "Links", "Slices", "Epochs",
        "Final Gap", "Runtime (s)", "TSTT Δ (%)",
    ]
    rows: dict[str, list] = {h: [] for h in header_cells}
    for nsr in results:
        for p in nsr.points:
            rows["Network"].append(nsr.spec.name)
            rows["Nodes"].append(nsr.spec.n_nodes)
            rows["Links"].append(nsr.spec.n_links)
            rows["Slices"].append(p.n_slices)
            rows["Epochs"].append(p.n_epochs)
            rows["Final Gap"].append(
                f"{p.final_gap:.6f}" if p.final_gap is not None else "—"
            )
            rows["Runtime (s)"].append(f"{p.total_time_s:.1f}")
            improve = (
                (p.greedy_tstt - p.final_tstt) / p.greedy_tstt * 100
                if p.greedy_tstt > 0 else 0.0
            )
            rows["TSTT Δ (%)"].append(f"{improve:+.2f}")

    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=header_cells,
            fill_color="#37474F",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[rows[h] for h in header_cells],
            fill_color=[
                ["#E3F2FD" if i % 2 == 0 else "white"
                 for i in range(len(rows["Network"]))]
            ],
            font=dict(size=11),
            align="center",
        ),
    )])
    fig_table.update_layout(
        title="Sweep Summary",
        height=80 + 30 * len(rows["Network"]),
    )
    figs.append(fig_table)
    descriptions.append(
        "<h2>Summary Table</h2>"
        "<p>Complete results for all network × slice-count combinations.</p>"
    )

    return figs, descriptions


def generate_slice_sweep_report(
    tmp_path: str | Path | None = None,
    output_path: str = "docs/plots/slice_convergence_sweep.html",
) -> Path:
    """Generate the cross-network slice convergence sweep report."""
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    else:
        tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    specs = _build_specs()
    results: list[NetworkSweepResult] = []
    for spec in specs:
        results.append(_run_sweep(spec, tmp_path))

    figs, descriptions = _build_report_sections(results)

    network_summary = ", ".join(
        f"{r.spec.name} ({r.spec.n_nodes}n/{r.spec.n_links}e)"
        for r in results
    )
    intro = (
        "<p>Cross-network sensitivity analysis of the <b>matrix-free hill-climber</b> "
        f"convergence behavior as a function of departure slice count. Networks: "
        f"{network_summary}. "
        f"Sweep: {SWEEP_SLICES} slices × {MAX_EPOCHS} max epochs, "
        f"gap threshold {GAP_THRESHOLD}.</p>"
    )

    _write_combined_report(
        title="Slice Convergence Sweep",
        intro=intro,
        figures=figs,
        descriptions=descriptions,
        path=Path(output_path),
    )
    return Path(output_path)
