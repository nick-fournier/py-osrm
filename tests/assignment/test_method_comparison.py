"""MSA vs Frank-Wolfe method comparison report.

Runs both convergence methods on Anaheim and Chicago Sketch, then produces
a single combined HTML report comparing convergence behaviour and final state.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from osrm.assignment.plots import (
    _add_mfd_section,
    _add_correlation_section,
    _write_combined_report,
)

from .hillclimber_validation import (
    HillClimberValidationCase,
    _convergence_series,
    run_hillclimber_case,
)

logger = logging.getLogger(__name__)

# ── colours ──────────────────────────────────────────────────────────
MSA_COLOR = "#1565C0"
FW_COLOR = "#E65100"


@dataclass
class _NetworkComparison:
    name: str
    msa_case: HillClimberValidationCase
    fw_case: HillClimberValidationCase
    meta: dict


# ── network runners ──────────────────────────────────────────────────

def _run_network(
    name: str,
    prepare_fn,
    copy_fn,
    trip_builder,
    state_patch_factory,
    tmp_path: Path,
    max_rounds: int,
    load_rate: float = 0.10,
) -> _NetworkComparison:
    """Run both MSA and FW on a single network."""
    base, meta = prepare_fn(tmp_path)

    common = dict(
        meta=meta,
        copy_fn=copy_fn,
        trip_builder=trip_builder,
        demand_scale=1.0,
        bin_width_s=3600.0,
        state_patch_factory=state_patch_factory,
        load_rate=load_rate,
        max_rounds=max_rounds,
        gap_threshold=0.001,
    )
    logger.info("Running %s with MSA...", name)
    msa_case = run_hillclimber_case(
        base_path=base,
        run_dir=tmp_path / "msa",
        method="msa",
        **common,
    )
    logger.info("Running %s with FW...", name)
    fw_case = run_hillclimber_case(
        base_path=base,
        run_dir=tmp_path / "fw",
        method="fw",
        **common,
    )
    return _NetworkComparison(name=name, msa_case=msa_case, fw_case=fw_case, meta=meta)


def _run_anaheim(tmp_path: Path, max_rounds: int) -> _NetworkComparison:
    from .test_anaheim import (
        _prepare_anaheim_network,
        _copy_clean_osrm,
        _build_hillclimber_trips,
        patch_lanes,
    )
    return _run_network(
        "Anaheim",
        _prepare_anaheim_network,
        _copy_clean_osrm,
        _build_hillclimber_trips,
        lambda meta: lambda state: patch_lanes(state, meta),
        tmp_path / "anaheim",
        max_rounds,
    )


def _run_chicago_sketch(tmp_path: Path, max_rounds: int) -> _NetworkComparison:
    from .test_chicago_sketch import (
        _prepare_chicago_network,
        _copy_clean_osrm,
        _build_hillclimber_trips,
        patch_lanes,
    )
    return _run_network(
        "Chicago Sketch",
        _prepare_chicago_network,
        _copy_clean_osrm,
        _build_hillclimber_trips,
        lambda meta: lambda state: patch_lanes(state, meta),
        tmp_path / "chi_sketch",
        max_rounds,
    )


# ── plot builders ────────────────────────────────────────────────────

def _moving_max(values: list[float], window: int = 5) -> list[float]:
    arr = np.array(values, dtype=float)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = arr[start : i + 1].max()
    return out.tolist()


def _add_convergence_comparison(
    figs: list,
    descriptions: list,
    comp: _NetworkComparison,
):
    """Convergence overlay: TSTT, gap, Δk for MSA vs FW on one network."""

    msa_batches = comp.msa_case.result.batch_results or []
    fw_batches = comp.fw_case.result.batch_results or []
    msa_ref = _convergence_series(comp.msa_case.result)
    fw_ref = _convergence_series(comp.fw_case.result)

    # ── TSTT ──
    fig_tstt = go.Figure()

    # Greedy phase (shared — same for both since same demand/load_rate)
    greedy_labels = [f"Load {i+1}" for i in range(len(msa_batches))]
    greedy_tstt = [b.network_tstt for b in msa_batches]
    n_greedy = len(greedy_labels)

    if greedy_tstt:
        fig_tstt.add_trace(go.Scatter(
            x=list(range(n_greedy)),
            y=greedy_tstt,
            mode="lines+markers",
            name="Greedy",
            line=dict(color="#888", width=2),
            marker=dict(size=4),
        ))

    # MSA refinement
    if msa_ref["network_tstt"]:
        xs = list(range(n_greedy, n_greedy + len(msa_ref["network_tstt"])))
        fig_tstt.add_trace(go.Scatter(
            x=xs, y=msa_ref["network_tstt"],
            mode="lines+markers", name="MSA",
            line=dict(color=MSA_COLOR, width=2),
            marker=dict(size=5),
        ))

    # FW refinement
    if fw_ref["network_tstt"]:
        xs = list(range(n_greedy, n_greedy + len(fw_ref["network_tstt"])))
        fig_tstt.add_trace(go.Scatter(
            x=xs, y=fw_ref["network_tstt"],
            mode="lines+markers", name="FW",
            line=dict(color=FW_COLOR, width=2),
            marker=dict(size=5),
        ))

    if n_greedy:
        fig_tstt.add_vline(x=n_greedy - 0.5, line_dash="dot", line_color="#666")

    fig_tstt.update_layout(
        title=f"{comp.name} — TSTT Convergence",
        xaxis_title="Step", yaxis_title="Network TSTT (veh·s)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    figs.append(fig_tstt)

    msa_final = msa_ref["network_tstt"][-1] if msa_ref["network_tstt"] else 0
    fw_final = fw_ref["network_tstt"][-1] if fw_ref["network_tstt"] else 0
    descriptions.append(
        f"<h3>{comp.name} — TSTT Convergence</h3>"
        f"<p>MSA final TSTT: <b>{msa_final:,.0f}</b>. "
        f"FW final TSTT: <b>{fw_final:,.0f}</b>.</p>"
    )

    # ── Relative Gap ──
    fig_gap = go.Figure()
    if msa_ref["gaps"]:
        abs_gaps = [abs(g) if g != 0 else None for g in msa_ref["gaps"]]
        xs = list(range(1, len(abs_gaps) + 1))
        fig_gap.add_trace(go.Scatter(
            x=xs, y=abs_gaps,
            mode="markers", name="MSA (raw)",
            marker=dict(color=MSA_COLOR, size=4, opacity=0.3),
        ))
        fig_gap.add_trace(go.Scatter(
            x=xs, y=_moving_max([abs(g) for g in msa_ref["gaps"]]),
            mode="lines", name="MSA (envelope)",
            line=dict(color=MSA_COLOR, width=1.5, dash="dash"),
        ))
    if fw_ref["gaps"]:
        abs_gaps = [abs(g) if g != 0 else None for g in fw_ref["gaps"]]
        xs = list(range(1, len(abs_gaps) + 1))
        fig_gap.add_trace(go.Scatter(
            x=xs, y=abs_gaps,
            mode="lines+markers", name="FW",
            line=dict(color=FW_COLOR, width=2),
            marker=dict(size=5),
        ))
    fig_gap.update_layout(
        title=f"{comp.name} — Wardrop Relative Gap",
        xaxis_title="Convergence Iteration", yaxis_title="|Relative Gap|",
        yaxis_type="log", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    figs.append(fig_gap)

    msa_final_gap = abs(msa_ref["gaps"][-1]) if msa_ref["gaps"] else float("nan")
    fw_final_gap = abs(fw_ref["gaps"][-1]) if fw_ref["gaps"] else float("nan")
    descriptions.append(
        f"<h3>{comp.name} — Wardrop Gap</h3>"
        f"<p>MSA final gap: <b>{msa_final_gap:.6f}</b>. "
        f"FW final gap: <b>{fw_final_gap:.6f}</b>. "
        "MSA uses fixed &alpha;=1/n schedule; FW uses Beckmann line search.</p>"
    )

    # ── Δk (state change norm) ──
    fig_dk = go.Figure()
    if msa_ref["state_change_norm"]:
        xs = list(range(1, len(msa_ref["state_change_norm"]) + 1))
        fig_dk.add_trace(go.Scatter(
            x=xs, y=msa_ref["state_change_norm"],
            mode="lines+markers", name="MSA",
            line=dict(color=MSA_COLOR, width=2),
            marker=dict(size=5),
        ))
    if fw_ref["state_change_norm"]:
        xs = list(range(1, len(fw_ref["state_change_norm"]) + 1))
        fig_dk.add_trace(go.Scatter(
            x=xs, y=fw_ref["state_change_norm"],
            mode="lines+markers", name="FW",
            line=dict(color=FW_COLOR, width=2),
            marker=dict(size=5),
        ))
    fig_dk.update_layout(
        title=f"{comp.name} — State Change Norm (Δk)",
        xaxis_title="Convergence Iteration", yaxis_title="‖Δk‖ / ‖k‖",
        yaxis_type="log", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    figs.append(fig_dk)
    descriptions.append(
        f"<h3>{comp.name} — State Change Norm</h3>"
        "<p>Relative change in link density vector per iteration. "
        "Monotonically decreasing by construction for MSA (&alpha;=1/n shrinks). "
        "FW step size is adaptive and may not shrink monotonically.</p>"
    )

    # ── Step size comparison ──
    n_msa = len(msa_ref["gaps"]) if msa_ref["gaps"] else 0
    n_fw = len(fw_ref["gaps"]) if fw_ref["gaps"] else 0
    n_max = max(n_msa, n_fw, 1)

    fig_step = go.Figure()
    msa_alphas = [1.0 / (m + 1) for m in range(1, n_max + 1)]
    fig_step.add_trace(go.Scatter(
        x=list(range(1, n_max + 1)), y=msa_alphas,
        mode="lines", name="MSA (1/n)",
        line=dict(color=MSA_COLOR, width=2, dash="dot"),
    ))

    # FW step sizes from MSA iteration results (they store alpha)
    fw_msa_results = getattr(comp.fw_case.result, "msa_results", []) or []
    if fw_msa_results:
        fw_alphas = []
        for r in fw_msa_results:
            alpha = getattr(r, "alpha", None)
            if alpha is None:
                alpha = 1.0 / (r.iteration + 1)
            fw_alphas.append(alpha)
        fig_step.add_trace(go.Scatter(
            x=list(range(1, len(fw_alphas) + 1)), y=fw_alphas,
            mode="lines+markers", name="FW (line search)",
            line=dict(color=FW_COLOR, width=2),
            marker=dict(size=5),
        ))

    fig_step.update_layout(
        title=f"{comp.name} — Step Size per Iteration",
        xaxis_title="Convergence Iteration", yaxis_title="Step Size (α)",
        template="plotly_white",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    figs.append(fig_step)
    descriptions.append(
        f"<h3>{comp.name} — Step Size</h3>"
        "<p>MSA uses a fixed 1/n schedule (dotted). FW uses Beckmann line search "
        "to find the optimal step size each iteration — large early, small near "
        "equilibrium.</p>"
    )


def _add_final_state_comparison(
    figs: list,
    descriptions: list,
    comp: _NetworkComparison,
):
    """MFD and correlation for both methods' final states."""
    msa_state = comp.msa_case.result.network_state
    fw_state = comp.fw_case.result.network_state
    link_attrs = comp.meta.get("link_attrs", {})
    ref = comp.meta.get("ref_flows", {})

    # Section header
    figs.append(None)
    descriptions.append(
        f"<h3>{comp.name} — Final State Comparison</h3>"
        "<p>MFD and correlation plots for both methods at their final converged states.</p>"
    )

    # MFD — MSA (the function adds its own h2, so we add a note before)
    _add_mfd_section(figs, descriptions, msa_state, 1.0)
    # Relabel the last description to clarify it's MSA
    descriptions[-1] = descriptions[-1].replace(
        "<h2>Macroscopic Fundamental Diagram (MFD)</h2>",
        f"<h2>{comp.name} — MFD (MSA)</h2>",
    )

    # MFD — FW
    _add_mfd_section(figs, descriptions, fw_state, 1.0)
    descriptions[-1] = descriptions[-1].replace(
        "<h2>Macroscopic Fundamental Diagram (MFD)</h2>",
        f"<h2>{comp.name} — MFD (FW)</h2>",
    )

    # Correlation (if reference flows available)
    if ref:
        _add_correlation_section(figs, descriptions, msa_state, ref, link_attrs, 1.0)
        descriptions[-1] = descriptions[-1].replace(
            "<h2>Correlation: MFD vs BPR</h2>",
            f"<h2>{comp.name} — Correlation (MSA)</h2>",
        )

        _add_correlation_section(figs, descriptions, fw_state, ref, link_attrs, 1.0)
        descriptions[-1] = descriptions[-1].replace(
            "<h2>Correlation: MFD vs BPR</h2>",
            f"<h2>{comp.name} — Correlation (FW)</h2>",
        )


# ── report generator ─────────────────────────────────────────────────

def generate_method_comparison_report(
    tmp_path: str | Path,
    output_path: str = "plots/method_comparison.html",
    max_rounds: int = 20,
) -> Path:
    """Generate MSA vs FW comparison report on Anaheim and Chicago Sketch."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    anaheim = _run_anaheim(tmp_path, max_rounds)
    chi_sketch = _run_chicago_sketch(tmp_path, max_rounds)

    figs: list[go.Figure | None] = []
    descriptions: list[str] = []

    for comp in [anaheim, chi_sketch]:
        figs.append(None)
        descriptions.append(
            f"<h2>{comp.name}</h2>"
            f"<p>MSA vs Frank-Wolfe convergence comparison on {comp.name} "
            f"({max_rounds} max iterations, 10 greedy load steps, 100% demand).</p>"
        )
        _add_convergence_comparison(figs, descriptions, comp)
        _add_final_state_comparison(figs, descriptions, comp)

    _write_combined_report(
        title="MSA vs Frank-Wolfe Method Comparison",
        intro=(
            "<p>Side-by-side convergence and final-state comparison of MSA "
            "(α=1/n fixed schedule) and Frank-Wolfe (Beckmann line search) "
            "on two TNTP benchmark networks.</p>"
            f"<p>Both methods use identical greedy loading as warm start, then "
            f"run up to <b>{max_rounds}</b> convergence iterations.</p>"
        ),
        figures=figs,
        descriptions=descriptions,
        path=Path(output_path),
    )
    return Path(output_path)
