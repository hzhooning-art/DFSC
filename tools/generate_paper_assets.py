"""Generate manuscript-ready tables and figures from experiment outputs.

The script intentionally uses only the Python standard library plus matplotlib.
It reads the checked-in CSV/JSON result files and writes publication assets into
``paper_assets`` at the repository root.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper1_mlsl" / "results"
TABLES = RESULTS / "tables"
ASSETS = ROOT / "paper_assets"
FIG_DIR = ASSETS / "figures"
TAB_DIR = ASSETS / "tables"
os.environ.setdefault("MPLCONFIGDIR", str(ASSETS / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(name: str) -> dict:
    with (RESULTS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def fnum(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-2 or abs_value >= 1e3:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_table(headers: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    if label not in {"tab:prior-tool-comparison", "tab:protocol-map", "tab:software-artifact"}:
        columns = "l" + "c" * (len(headers) - 1)
        safe_headers = [latex_escape(h) for h in headers]
        safe_rows = [[latex_escape(cell) for cell in row] for row in rows]
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{latex_escape(caption)}}}",
            f"\\label{{{label}}}",
            "\\resizebox{\\linewidth}{!}{%",
            f"\\begin{{tabular}}{{{columns}}}",
            "\\toprule",
            " & ".join(safe_headers) + " \\\\",
            "\\midrule",
        ]
        for row in safe_rows:
            lines.append(" & ".join(row) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
        return "\n".join(lines)

    safe_headers = [latex_escape(h) for h in headers]
    safe_rows = [[latex_escape(cell) for cell in row] for row in rows]
    if label in {"tab:prior-tool-comparison", "tab:protocol-map", "tab:software-artifact"}:
        special_widths = {
            "tab:prior-tool-comparison": ["0.17", "0.25", "0.28", "0.22"],
            "tab:protocol-map": ["0.23", "0.34", "0.34"],
            "tab:software-artifact": ["0.18", "0.24", "0.22", "0.27"],
        }
        widths = special_widths[label]
        columns = "@{}" + "".join(f"p{{{width}\\linewidth}}" for width in widths) + "@{}"
    else:
        width_map = {
            1: ["0.92"],
            2: ["0.28", "0.64"],
            3: ["0.22", "0.34", "0.34"],
            4: ["0.16", "0.24", "0.26", "0.26"],
            5: ["0.13", "0.19", "0.25", "0.16", "0.19"],
        }
        widths = width_map.get(len(headers), ["0.13"] * len(headers))
        columns = "@{}" + "".join(f"p{{{width}\\linewidth}}" for width in widths) + "@{}"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(safe_headers) + " \\\\",
        "\\midrule",
    ]
    for row in safe_rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def summarize_baselines() -> tuple[list[list[str]], list[str], list[float], list[float]]:
    rows = read_csv("larger_neural_baseline_multiseed.csv")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(float(row["long_time_relative_error"]))

    table_rows: list[list[str]] = []
    names: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for model in sorted(grouped):
        values = grouped[model]
        names.append(model)
        means.append(mean(values))
        stds.append(sample_std(values))
        table_rows.append([model, str(len(values)), fnum(mean(values)), fnum(sample_std(values))])
    return table_rows, names, means, stds


def summarize_fpinn() -> tuple[list[list[str]], dict[str, float]]:
    rows = read_csv("fpinn_repeated_baseline.csv")
    alpha_errors = [float(row["alpha_relative_error"]) for row in rows]
    solution_errors = [float(row["solution_relative_error"]) for row in rows]
    stats = {
        "alpha_mean": mean(alpha_errors),
        "alpha_std": sample_std(alpha_errors),
        "solution_mean": mean(solution_errors),
        "solution_std": sample_std(solution_errors),
    }
    table_rows = [
        [
            "fPINN scalar inverse",
            str(len(rows)),
            f"{fnum(stats['alpha_mean'])} +/- {fnum(stats['alpha_std'])}",
            f"{fnum(stats['solution_mean'])} +/- {fnum(stats['solution_std'])}",
        ]
    ]
    return table_rows, stats


def summarize_pde_field_fpinn() -> tuple[list[list[str]], dict[str, float]]:
    rows = read_csv("pde_field_fpinn_baseline.csv")
    alpha_errors = [float(row["alpha_relative_error"]) for row in rows]
    solution_errors = [float(row["solution_relative_error"]) for row in rows]
    residuals = [float(row["residual_rms"]) for row in rows]
    stats = {
        "alpha_mean": mean(alpha_errors),
        "alpha_std": sample_std(alpha_errors),
        "solution_mean": mean(solution_errors),
        "solution_std": sample_std(solution_errors),
        "residual_mean": mean(residuals),
    }
    table_rows = [
        [
            "PDE-field fPINN",
            str(len(rows)),
            f"{fnum(stats['alpha_mean'])} +/- {fnum(stats['alpha_std'])}",
            f"{fnum(stats['solution_mean'])} +/- {fnum(stats['solution_std'])}",
            fnum(stats["residual_mean"]),
        ]
    ]
    return table_rows, stats


def summarize_hybrid_backbone() -> list[list[str]]:
    rows = read_csv("hybrid_backbone_baseline.csv")
    grouped: dict[str, list[float]] = defaultdict(list)
    train_grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(float(row["long_time_relative_error"]))
        train_grouped[row["model"]].append(float(row["train_relative_error"]))

    order = [
        "MLSL_backbone_only",
        "Hybrid_MLSL_residual",
        "DeepONet_latent64_hidden96",
        "FNO1D_width32_modes12",
    ]
    table_rows: list[list[str]] = []
    for model in order:
        values = grouped[model]
        train_values = train_grouped[model]
        table_rows.append(
            [
                model,
                str(len(values)),
                f"{fnum(mean(train_values))}",
                f"{fnum(mean(values))} +/- {fnum(sample_std(values))}",
            ]
        )
    return table_rows


def summarize_downstream_forcing() -> list[list[str]]:
    rows = read_csv("downstream_forcing_composition.csv")
    grouped: dict[str, list[float]] = defaultdict(list)
    train_grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(float(row["long_time_relative_error"]))
        train_grouped[row["model"]].append(float(row["train_relative_error"]))

    order = [
        "MLSL_backbone_only",
        "Hybrid_MLSL_unknown_forcing",
        "FNO1D_width36_modes12",
        "DeepONet_latent72_hidden112",
    ]
    table_rows: list[list[str]] = []
    for model in order:
        values = grouped[model]
        train_values = train_grouped[model]
        table_rows.append(
            [
                model,
                str(len(values)),
                f"{fnum(mean(train_values))}",
                f"{fnum(mean(values))} +/- {fnum(sample_std(values))}",
            ]
        )
    return table_rows


def summarize_transition_inverse_baseline() -> list[list[str]]:
    transition = read_json("transition_and_inverse_baseline_summary.json")
    inverse_rows = read_csv("same_setting_inverse_baseline_summary.csv")
    table_rows: list[list[str]] = [
        [
            "Transition probe",
            str(transition["transition_probe_cases"]),
            f"finite value/grad {fnum(transition['transition_finite_value_rate'])}/{fnum(transition['transition_finite_grad_rate'])}",
            f"max adjacent value/grad jump {fnum(transition['transition_max_adjacent_value_jump'])}/{fnum(transition['transition_max_adjacent_grad_jump'])}",
        ]
    ]
    for row in inverse_rows:
        table_rows.append(
            [
                row["model"],
                row["seeds"],
                f"alpha err. mean/max {fnum(float(row['alpha_error_mean']))}/{fnum(float(row['alpha_error_max']))}",
                f"solution err. mean/max {fnum(float(row['solution_error_mean']))}/{fnum(float(row['solution_error_max']))}",
            ]
        )
    return table_rows


def summarize_contract_audit() -> list[list[str]]:
    rows = read_csv("primitive_contract_audit.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["axis"]].append(row)

    labels = {
        "shape_autograd": "Shape/autograd contract",
        "constructor": "Boundary constructors",
        "gpu_precision": "GPU and reduced precision",
    }
    meanings = {
        "shape_autograd": "Single/batched tensor semantics and alpha/beta gradients",
        "constructor": "1D/2D Dirichlet, Neumann, periodic, and mixed bases",
        "gpu_precision": "CUDA batched forward with float64/float32/float16/bfloat16",
    }
    table_rows: list[list[str]] = []
    for axis in ["shape_autograd", "constructor", "gpu_precision"]:
        axis_rows = grouped[axis]
        passed = sum(1 for row in axis_rows if row["passed"].lower() == "true")
        total = len(axis_rows)
        table_rows.append(
            [
                labels[axis],
                f"{passed}/{total} passed",
                ", ".join(sorted({row["dtype"] for row in axis_rows})),
                meanings[axis],
            ]
        )
    return table_rows


def summarize_software_artifact() -> list[list[str]]:
    rows = read_csv("software_artifact_audit.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["axis"]].append(row)

    labels = {
        "dfsc_ecosystem": "dfsc ecosystem",
        "api": "Public API",
        "packaging": "Packaging",
        "documentation": "Documentation",
        "examples": "Examples",
        "tests": "Tests",
        "diagnostics": "Diagnostics",
        "reproducibility": "Reproducibility",
        "results": "Result assets",
        "paper_assets": "Paper assets",
    }
    meanings = {
        "dfsc_ecosystem": "Single package namespace, registry, adapters, and workflows",
        "api": "Layer, evaluator, baseline, and diagnostic symbols",
        "packaging": "Editable package metadata and dependency record",
        "documentation": "API, status, and validation notes",
        "examples": "Quickstart, batched, and inverse scripts",
        "tests": "Layer, evaluator, and forced-wrapper tests",
        "diagnostics": "Doctor, maturity, and reproduction utilities",
        "reproducibility": "Validation and reproduction entry points",
        "results": "CSV/JSON outputs used by the manuscript",
        "paper_assets": "Regenerated tables and figures",
    }
    order = [
        "dfsc_ecosystem",
        "api",
        "packaging",
        "documentation",
        "examples",
        "tests",
        "diagnostics",
        "reproducibility",
        "results",
        "paper_assets",
    ]
    table_rows: list[list[str]] = []
    for axis in order:
        axis_rows = grouped.get(axis, [])
        if not axis_rows:
            continue
        passed = sum(1 for row in axis_rows if row["passed"].lower() == "true")
        total = len(axis_rows)
        observed = "; ".join(row["observed"] for row in axis_rows[:2])
        if len(axis_rows) > 2:
            observed += f"; +{len(axis_rows) - 2} checks"
        observed = (
            observed.replace("editable Python package metadata", "pyproject")
            .replace("minimal dependency record", "requirements")
            .replace("API reference page", "API doc")
            .replace("implementation status page", "status doc")
            .replace("Python examples", "examples")
            .replace("unit-test files", "test files")
            .replace("tool scripts", "tools")
            .replace("CSV result tables", "CSV tables")
            .replace("JSON summaries", "JSON")
            .replace("generated LaTeX tables", "LaTeX tables")
            .replace("generated PDF figures", "PDF figures")
            .replace("tools/reproduce_core.py", "reproduce core")
            .replace("validate.py", "validate")
        )
        table_rows.append([labels[axis], f"{passed}/{total} passed", observed, meanings[axis]])
    return table_rows


def summarize_gpu_batch() -> list[list[str]]:
    rows = read_csv("gpu_batch_profile.csv")
    table_rows: list[list[str]] = []
    for dtype in ["float64", "float32"]:
        selected = [row for row in rows if row["dtype"] == dtype and row["batch_size"] in {"1", "1024"}]
        selected.sort(key=lambda row: int(row["batch_size"]))
        if len(selected) != 2:
            continue
        b1, b1024 = selected
        table_rows.append(
            [
                dtype,
                fnum(float(b1["seconds_min"])),
                fnum(float(b1024["seconds_min"])),
                fnum(float(b1024["seconds_per_sample"])),
                b1024["output_shape"],
            ]
        )
    return table_rows


def count_passed(rows: list[dict[str, str]], fields: list[str]) -> tuple[int, int]:
    count = sum(1 for row in rows if all(str(row[field]).lower() == "true" for field in fields))
    return count, len(rows)


def build_generality_rows() -> list[list[str]]:
    primitive_ok, primitive_total = count_passed(
        read_csv("primitive_generality_matrix.csv"), ["finite_output", "finite_alpha_grad"]
    )
    boundary_ok, boundary_total = count_passed(
        read_csv("boundary_generality.csv"), ["finite_output", "finite_alpha_grad", "finite_beta_grad"]
    )
    two_d_ok, two_d_total = count_passed(
        read_csv("two_d_boundary_extension.csv"), ["finite_output", "finite_alpha_grad", "finite_beta_grad"]
    )
    reaction_ok, reaction_total = count_passed(
        read_csv("reaction_diffusion_family.csv"), ["finite_output", "finite_alpha_grad", "finite_beta_grad"]
    )
    semilinear_ok, semilinear_total = count_passed(
        read_csv("semilinear_backbone_extension.csv"), ["finite_output", "finite_alpha_grad", "finite_beta_grad"]
    )
    forcing_q128 = [
        float(row["relative_error"])
        for row in read_csv("manufactured_forcing_validation.csv")
        if row["quadrature_points"] == "128"
    ]

    return [
        ["Layer API", "1D/batch/order/device checks", f"{primitive_ok}/{primitive_total} passed", "Core layer coverage"],
        ["Boundary constructors", "Dirichlet/Neumann/periodic/mixed", f"{boundary_ok}/{boundary_total} passed", "Reusable basis coverage"],
        ["2D boundaries", "Neumann/periodic/mixed 2D", f"{two_d_ok}/{two_d_total} passed", "Beyond 1D examples"],
        ["Reaction-diffusion", "shifted spectral rates", f"{reaction_ok}/{reaction_total} passed", "Linear PDE-family extension"],
        ["Semilinear backbone", "cubic correction wrapper", f"{semilinear_ok}/{semilinear_total} passed", "Hybrid workflow smoke test"],
        ["Manufactured forcing", "q=128 Duhamel validation", f"mean {fnum(mean(forcing_q128))}", "Nonhomogeneous dynamics"],
    ]


def build_protocol_rows() -> list[list[str]]:
    return [
        ["Autograd and trainable orders", "Finite-difference checks; inverse recovery; noisy/sparse ablations", "Gradient errors; mean/std order errors"],
        ["Evaluator reliability", "Reference grid; negative-real tail; transition-neighborhood probes", "Value error; finite rates; gradient continuity diagnostics"],
        ["Layer interface", "Shape, constructor, boundary, batch, and GPU contract audits", "Pass matrix; CPU/GPU consistency; batch profile"],
        ["Use advantage", "L1 history comparison; matched fPINN inverse; FNO/DeepONet diagnostics", "Speedup; inverse error; long-time neural errors"],
        ["Composition", "MLSL backbone with residual or unknown-forcing neural heads", "Train and long-time relative errors"],
        ["Software artifact", "API, packaging, examples, tests, diagnostics, result assets, and paper assets", "Artifact-audit pass matrix"],
    ]


def build_prior_tool_rows() -> list[list[str]]:
    return [
        ["Classical fractional solvers", "High-accuracy history marching and numerical analysis", "Usually solver-step interface; orders often external settings", "Not the target; MLSL supplies a layer interface for known spectral propagators"],
        ["Differentiable equation ecosystems", "Mature ODE/PDE time-stepping and adjoint workflows", "Fractional memory and Mittag-Leffler spectral propagation are usually represented through solver state rather than a compact layer in this artifact", "Complementary infrastructure; MLSL targets the missing direct-query spectral layer"],
        ["fPINN / residual training", "Flexible residual-based inverse and forward learning", "Learns neural field and residual; not a direct propagator layer", "Complementary; MLSL is advantageous when the spectral propagator is known"],
        ["FNO / DeepONet", "Data-driven operator approximation", "May relearn known fractional propagation from samples", "Used as diagnostic controls in known-propagator settings"],
        ["NeuralOperator-style libraries", "Reusable neural-operator training ecosystem", "No standardized Mittag-Leffler spectral primitive with trainable alpha/beta in this artifact", "Future integration target rather than a dependency"],
        ["dfsc", "Known Mittag-Leffler spectral evolution exposed as a dfsc primitive", "Restricted to tested spectral regimes and real non-positive arguments", "Batched, differentiable, GPU-capable, and composable component ecosystem"],
    ]


def build_baseline_budget_rows() -> list[list[str]]:
    return [
        ["MLSL inverse", "exp03/05/32/33", "Adam on bounded alpha/beta", "250--500 steps", "Synthetic spectral observations; float64"],
        ["PDE-field fPINN", "exp28/33", "MLPField hidden 64 depth 3 + L1 Caputo residual", "700 steps, 3 seeds", "Matched subdiffusive field inverse setting"],
        ["Hybrid MLSL residual", "exp29", "MLSL backbone + small residual head", "3 seeds", "Known backbone with unmodeled correction"],
        ["Unknown-forcing hybrid", "exp31", "MLSL backbone + forcing correction head", "3 seeds", "Downstream synthetic forcing task"],
        ["FNO / DeepONet controls", "exp22/29/31", "Width/modes or latent/hidden settings reported in model names", "3 seeds", "Diagnostic controls; not exhaustive benchmark sweeps"],
        ["GPU/batch checks", "exp27/30", "RTX 5070; PyTorch 2.11.0+cu128; CUDA 12.8", "Repeated synchronized forward checks", "Finite gradients and throughput diagnostics"],
        ["Evaluator switch ablation", "exp34", "Hard switch vs smooth transition", "246 transition-neighborhood probes", "Value/gradient jump diagnostics"],
        ["Software artifact audit", "exp35", "API/docs/examples/tests/reproducibility/assets", "Static artifact checks", "Software-paper maturity evidence"],
    ]


def summarize_switch_ablation() -> list[list[str]]:
    rows = read_csv("evaluator_switch_ablation_summary.csv")
    out: list[list[str]] = []
    for row in rows:
        out.append(
            [
                row["method"],
                row["alpha"],
                row["threshold"],
                row["cases"],
                fnum(float(row["max_adjacent_value_jump"])),
                fnum(float(row["max_adjacent_grad_jump"])),
            ]
        )
    return out


def build_tables() -> dict[str, str]:
    summary = read_json("summary.json")
    gap = read_json("gap_closure_summary.json")
    gpu = read_json("gpu_validation_summary.json")
    paper_grade = read_json("paper_grade_extension_summary.json")
    hybrid_rows = summarize_hybrid_backbone()
    downstream_rows = summarize_downstream_forcing()
    transition_inverse_rows = summarize_transition_inverse_baseline()
    generality_rows = build_generality_rows()
    protocol_rows = build_protocol_rows()
    prior_tool_rows = build_prior_tool_rows()
    baseline_budget_rows = build_baseline_budget_rows()
    contract_rows = summarize_contract_audit()
    gpu_batch_rows = summarize_gpu_batch()
    switch_ablation_rows = summarize_switch_ablation()
    software_rows = summarize_software_artifact()
    contract_summary = read_json("primitive_contract_audit_summary.json")
    switch_summary = read_json("evaluator_switch_ablation_summary.json")
    software_summary = read_json("software_artifact_audit_summary.json")

    evidence_rows = [
        ["Differentiable orders", "alpha/beta gradient checks", f"{fnum(summary['alpha_grad_min_relative_error'])} / {fnum(summary['beta_grad_min_relative_error'])}", "Valid autograd path"],
        ["Inverse recovery", "5 random seeds", f"{fnum(paper_grade['multi_seed_alpha_error_mean'])} / {fnum(paper_grade['multi_seed_beta_error_mean'])}", "Learnable fractional orders"],
        ["History-free query", "L1 comparison at 400 steps", f"{fnum(paper_grade['proposition_history_free_speedup_at_400'])}x", "Direct query advantage"],
        ["Boundary generality", "Boundary matrix", "45/45 passed", "Reusable basis abstraction"],
        ["Two-parameter kernel", "High-precision reference", fnum(gap["eab_reference_max_relative_error"]), "Forced dynamics support"],
        ["Manufactured forcing", "q=128 validation", "mean 3.05e-03", "Nonhomogeneous workflow"],
        ["GPU execution", "RTX 5070 CPU/GPU check", fnum(gpu["gpu_consistency_max_output_relative_error"]), "Batched tensor layer"],
        ["Layer contract", "shape/constructor/GPU audit", f"{contract_summary['passed_checks']}/{contract_summary['total_checks']} passed", "Layer-interface support"],
        ["Transition and inverse baseline", "63 transition probes + same-setting fPINN comparison", transition_inverse_rows[1][2], "Stable switch and direct inverse advantage"],
        ["Switch ablation", "hard vs smooth transition", f"{fnum(switch_summary['grad_jump_reduction_factor'])}x worst-gradient reduction", "Switching rule diagnostics"],
        ["Hybrid backbone", "3 residual-head seeds", hybrid_rows[1][3], "Composable MLSL layer"],
        ["Downstream forcing", "known backbone + unknown correction", downstream_rows[1][3], "Hybrid composition task"],
        ["Software environment", "artifact audit", f"{software_summary['passed_checks']}/{software_summary['total_checks']} passed", "Reusable package and reproduction workflow"],
    ]

    baseline_rows, _, _, _ = summarize_baselines()

    speed_rows = []
    for row in read_csv("proposition_evidence.csv"):
        speed_rows.append([
            row["num_steps"],
            fnum(float(row["mlsl_seconds"])),
            fnum(float(row["l1_seconds"])),
            fnum(float(row["speedup_l1_over_mlsl"])),
        ])

    gpu_rows = []
    for row in read_csv("gpu_cpu_consistency.csv"):
        gpu_rows.append([
            row["dtype"],
            fnum(float(row["output_relative_error"])),
            fnum(float(row["alpha_grad_relative_error"])),
            fnum(float(row["beta_grad_relative_error"])),
            row["finite_gpu_output"],
        ])

    assets = {
        "table_evidence.md": markdown_table(
            ["Claim", "Experiment", "Main result", "Interpretation"], evidence_rows
        ),
        "table_baselines.md": markdown_table(
            ["Model", "Seeds", "Long-time error mean", "Long-time error std"], baseline_rows
        ),
        "table_history_free.md": markdown_table(
            ["Steps", "MLSL seconds", "L1 seconds", "Speedup"], speed_rows
        ),
        "table_gpu.md": markdown_table(
            ["dtype", "Output rel. err.", "Alpha-grad rel. err.", "Beta-grad rel. err.", "Finite"], gpu_rows
        ),
        "table_hybrid_backbone.md": markdown_table(
            ["Model", "Seeds", "Train error", "Long-time error"], hybrid_rows
        ),
        "table_downstream_forcing.md": markdown_table(
            ["Model", "Seeds", "Train error", "Long-time error"], downstream_rows
        ),
        "table_transition_inverse.md": markdown_table(
            ["Block", "Cases/seeds", "Primary result", "Secondary result"], transition_inverse_rows
        ),
        "table_switch_ablation.md": markdown_table(
            ["Method", "Alpha", "Threshold", "Cases", "Max value jump", "Max alpha-grad jump"], switch_ablation_rows
        ),
        "table_primitive_contract.md": markdown_table(
            ["Contract axis", "Result", "Dtypes", "Meaning"], contract_rows
        ),
        "table_software_artifact.md": markdown_table(
            ["Axis", "Result", "Observed", "Meaning"], software_rows
        ),
        "table_gpu_batch.md": markdown_table(
            ["dtype", "Batch 1 sec.", "Batch 1024 sec.", "Batch 1024 sec/sample", "Output shape"], gpu_batch_rows
        ),
        "table_generality.md": markdown_table(
            ["Axis", "Experiment", "Result", "Meaning"], generality_rows
        ),
        "table_protocol.md": markdown_table(
            ["Evidence block", "Design", "Reported output"], protocol_rows
        ),
        "table_prior_tools.md": markdown_table(
            ["Approach", "Primary strength", "Interface limitation for this paper", "Relation to MLSL"], prior_tool_rows
        ),
        "table_baseline_budget.md": markdown_table(
            ["Block", "Scripts", "Model / optimization", "Budget", "Scope"], baseline_budget_rows
        ),
        "table_evidence.tex": latex_table(
            ["Claim", "Experiment", "Main result", "Interpretation"],
            evidence_rows,
            "Evidence map for the main MLSL neural-layer claims.",
            "tab:evidence-map",
        ),
        "table_baselines.tex": latex_table(
            ["Model", "Seeds", "Mean error", "Std"],
            baseline_rows,
            "Diagnostic long-time neural baseline comparison in the known-propagator setting. Std denotes sample standard deviation across seeds.",
            "tab:baseline-comparison",
        ),
        "table_history_free.tex": latex_table(
            ["Steps", "MLSL sec.", "L1 sec.", "Speedup"],
            speed_rows,
            "History-free query speed compared with L1 marching.",
            "tab:history-free-speed",
        ),
        "table_gpu.tex": latex_table(
            ["dtype", "Output err.", "Alpha-grad err.", "Beta-grad err.", "Finite"],
            gpu_rows,
            "CPU/GPU consistency on RTX 5070.",
            "tab:gpu-consistency",
        ),
        "table_hybrid_backbone.tex": latex_table(
            ["Model", "Seeds", "Train error", "Long-time error"],
            hybrid_rows,
            "Hybrid MLSL-backbone residual head versus pure neural baselines. Reported uncertainties are sample standard deviations across seeds.",
            "tab:hybrid-backbone",
        ),
        "table_downstream_forcing.tex": latex_table(
            ["Model", "Seeds", "Train error", "Long-time error"],
            downstream_rows,
            "Downstream unknown-forcing composition task. The hybrid model inserts the MLSL backbone and learns only the correction.",
            "tab:downstream-forcing",
        ),
        "table_transition_inverse.tex": latex_table(
            ["Block", "Cases/seeds", "Primary result", "Secondary result"],
            transition_inverse_rows,
            "Transition-neighborhood evaluator diagnostics and same-setting inverse comparison with a PDE-field fPINN baseline.",
            "tab:transition-inverse",
        ),
        "table_switch_ablation.tex": latex_table(
            ["Method", "Alpha", "Threshold", "Cases", "Max value jump", "Max alpha-grad jump"],
            switch_ablation_rows,
            "Hard-switch versus smooth-transition ablation for the one-parameter Mittag-Leffler evaluator near branch thresholds.",
            "tab:switch-ablation",
        ),
        "table_primitive_contract.tex": latex_table(
            ["Contract axis", "Result", "Dtypes", "Meaning"],
            contract_rows,
            "Layer-interface audit for MLSL as a selectable differentiable spectral layer.",
            "tab:primitive-contract",
        ),
        "table_software_artifact.tex": latex_table(
            ["Axis", "Result", "Observed", "Meaning"],
            software_rows,
            "Software-artifact audit for the dfsc environment. The table checks the reusable API, documentation, examples, tests, diagnostics, result files, and manuscript-generation assets used by this paper.",
            "tab:software-artifact",
        ),
        "table_gpu_batch.tex": latex_table(
            ["dtype", "Batch 1 sec.", "Batch 1024 sec.", "Batch 1024 sec/sample", "Output shape"],
            gpu_batch_rows,
            "RTX 5070 batch scaling profile for MLSL forward evaluation.",
            "tab:gpu-batch-profile",
        ),
        "table_generality.tex": latex_table(
            ["Axis", "Experiment", "Result", "Meaning"],
            generality_rows,
            "Generality checks beyond the minimal homogeneous 1D setting.",
            "tab:generality-extension",
        ),
        "table_protocol.tex": latex_table(
            ["Evidence block", "Design", "Reported output"],
            protocol_rows,
            "Experiment protocol map used to align evidence with the MLSL differentiable-layer claim.",
            "tab:protocol-map",
        ),
        "table_prior_tools.tex": latex_table(
            ["Approach", "Primary strength", "Interface limitation", "Relation to MLSL"],
            prior_tool_rows,
            "Capability-oriented comparison with related approaches. The table distinguishes solver accuracy, residual learning, neural-operator approximation, and the dfsc component-ecosystem contribution targeted here.",
            "tab:prior-tool-comparison",
        ),
        "table_baseline_budget.tex": latex_table(
            ["Block", "Scripts", "Model / optimization", "Budget", "Scope"],
            baseline_budget_rows,
            "Baseline and validation budget summary. The neural comparisons are diagnostic controls in controlled synthetic settings rather than exhaustive benchmark sweeps.",
            "tab:baseline-budget",
        ),
    }

    return assets


def save_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })

    prop_rows = read_csv("proposition_evidence.csv")
    steps = [int(r["num_steps"]) for r in prop_rows]
    speedups = [float(r["speedup_l1_over_mlsl"]) for r in prop_rows]

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(steps, speedups, marker="o", linewidth=2, color="#2f6f9f")
    ax.set_xlabel("Number of time steps")
    ax.set_ylabel("L1 time / MLSL time")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_history_free_speedup.png")
    fig.savefig(FIG_DIR / "fig_history_free_speedup.pdf")
    plt.close(fig)

    inv_rows = read_csv("multi_seed_inverse.csv")
    seeds = [int(r["seed"]) for r in inv_rows]
    alpha_err = [float(r["alpha_relative_error"]) for r in inv_rows]
    beta_err = [float(r["beta_relative_error"]) for r in inv_rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(seeds, alpha_err, marker="o", label="alpha", color="#4d8f6f")
    ax.plot(seeds, beta_err, marker="s", label="beta", color="#aa5d5d")
    ax.set_yscale("log")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Relative error")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_inverse_recovery_multiseed.png")
    fig.savefig(FIG_DIR / "fig_inverse_recovery_multiseed.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.axis("off")
    boxes = [
        (0.03, 0.58, 0.18, 0.24, "Initial field\n$u_0(x)$"),
        (0.28, 0.58, 0.22, 0.24, "MLSL layer\n$E_\\alpha(-\\mu t^\\alpha)$"),
        (0.58, 0.58, 0.18, 0.24, "Backbone\n$u_{ML}(x,t)$"),
        (0.58, 0.18, 0.18, 0.24, "Neural head\n$residual$"),
        (0.82, 0.38, 0.15, 0.24, "Prediction\n$\\hat u(x,t)$"),
    ]
    for x0, y0, w, h, text in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                w,
                h,
                facecolor="#f4f7fb",
                edgecolor="#355c7d",
                linewidth=1.4,
                transform=ax.transAxes,
            )
        )
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", transform=ax.transAxes)
    arrows = [
        ((0.21, 0.70), (0.28, 0.70)),
        ((0.50, 0.70), (0.58, 0.70)),
        ((0.76, 0.70), (0.82, 0.50)),
        ((0.67, 0.58), (0.67, 0.42)),
        ((0.76, 0.30), (0.82, 0.50)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"},
        )
    ax.text(
        0.31,
        0.31,
        "$\\alpha,\\beta$ are trainable; batched tensors\nremain device-local",
        ha="center",
        va="center",
        fontsize=8.5,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_mlsl_computational_graph.png")
    fig.savefig(FIG_DIR / "fig_mlsl_computational_graph.pdf")
    plt.close(fig)


def build_asset_index(table_files: dict[str, str]) -> str:
    figure_lines = [
        "- `figures/fig_history_free_speedup.(png|pdf)`: L1 marching versus MLSL direct-query speedup.",
        "- `figures/fig_inverse_recovery_multiseed.(png|pdf)`: alpha/beta inverse recovery across seeds.",
        "- `figures/fig_mlsl_computational_graph.(png|pdf)`: MLSL as a composable differentiable spectral-evolution layer.",
    ]

    lines = [
        "# Paper Tables and Figures",
        "",
        "This file is generated by `paper1_mlsl/tools/generate_paper_assets.py`.",
        "",
        "## Figures",
        "",
        *figure_lines,
        "",
        "## Markdown Tables",
        "",
    ]
    for name in sorted(k for k in table_files if k.endswith(".md")):
        lines.append(f"### {name}")
        lines.append("")
        lines.append(table_files[name])
        lines.append("")

    lines.append("## LaTeX Tables")
    lines.append("")
    for name in sorted(k for k in table_files if k.endswith(".tex")):
        lines.append(f"- `tables/{name}`")

    return "\n".join(lines)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    table_files = build_tables()
    for name, text in table_files.items():
        save_text(TAB_DIR / name, text)

    save_figures()
    save_text(ASSETS / "TABLES_AND_FIGURES.md", build_asset_index(table_files))
    print(f"Wrote paper assets to {ASSETS}")


if __name__ == "__main__":
    main()
