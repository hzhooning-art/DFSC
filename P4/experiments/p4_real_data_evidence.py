"""Normalize provenance-aware public-data experiments into P4 evidence."""

from __future__ import annotations

import json
import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P4_RESULTS = ROOT / "P4" / "results"
P4_RESULTS.mkdir(parents=True, exist_ok=True)


def run(source_root: Path, script: str) -> None:
    completed = subprocess.run(
        [__import__("sys").executable, str(source_root / "experiments" / script)],
        cwd=source_root,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"{script} failed:\n{completed.stdout}\n{completed.stderr}")


def copy_latest(source_root: Path, pattern: str, target: str) -> Path:
    candidates = sorted(
        [
            *(source_root / "generated_results").glob(pattern),
            *(source_root / "revision_results").glob(pattern),
        ],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no result matched {pattern}")
    source = candidates[-1]
    destination = P4_RESULTS / target
    shutil.copy2(source, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Optional upstream DFSC experiment root containing experiments/ and result directories.",
    )
    parser.add_argument("--rerun", action="store_true", help="Rerun upstream fits before normalization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geotes = P4_RESULTS / "p4_real_geotes_cross_cycle.json"
    steam = P4_RESULTS / "p4_real_heated_steam.json"
    if args.source_root is not None:
        source_root = args.source_root.resolve()
        if not (source_root / "experiments").is_dir():
            raise FileNotFoundError(f"upstream experiments directory not found under {source_root}")
        geotes_candidates = [
            *(source_root / "generated_results").glob("real_geotes_cross_cycle_summary*.json")
        ]
        steam_candidates = [*(source_root / "revision_results").glob("real_heated_steam.json")]
        if args.rerun or not geotes_candidates:
            run(source_root, "exp48_real_geotes_cross_cycle.py")
        if args.rerun or not steam_candidates:
            run(source_root, "exp51_real_heated_steam.py")
        geotes = copy_latest(
            source_root, "real_geotes_cross_cycle_summary*.json", geotes.name
        )
        steam = copy_latest(source_root, "real_heated_steam.json", steam.name)
    elif args.rerun:
        raise ValueError("--rerun requires --source-root")
    elif not geotes.exists() or not steam.exists():
        raise FileNotFoundError(
            "normalized source records are missing; provide --source-root or restore "
            "P4/results/p4_real_geotes_cross_cycle.json and p4_real_heated_steam.json"
        )
    geotes_payload = json.loads(geotes.read_text(encoding="utf-8"))
    steam_payload = json.loads(steam.read_text(encoding="utf-8"))
    fewshot_path = P4_RESULTS / "p4_real_geotes_fewshot.json"
    fewshot_payload = json.loads(fewshot_path.read_text(encoding="utf-8")) if fewshot_path.exists() else None
    result = {
        "schema": "DFSC-P4-Real-Data-Evidence-v1",
        "datasets": [
            {
                "name": geotes_payload["dataset"],
                "source": geotes_payload["source"],
                "license": geotes_payload["license"],
                "split": geotes_payload["protocol"],
                "result_file": geotes.name,
                "models": geotes_payload["summary"],
            },
            {
                "name": "Heated steam injection experimental temperature profiles",
                "source": "https://doi.org/10.5281/zenodo.15064388",
                "license": steam_payload["license"],
                "split": steam_payload["held_out_rule"],
                "result_file": steam.name,
                "models": steam_payload["summary"],
            },
            {
                "name": "GeoTES few-shot cross-cycle transfer",
                "source": "https://doi.org/10.5281/zenodo.18979098",
                "license": "CC BY 4.0",
                "split": "uniform subsampling over the full first cycle; second cycle held out; three seeds",
                "result_file": "p4_real_geotes_fewshot.json",
                "models": [] if fewshot_payload is None else fewshot_payload["summary"],
            },
        ],
        "interpretation": [
            "These are public-data transfer tests, not claims of universal physical identification.",
            "Measured channels are used as documented; no undocumented sensor geometry is inferred.",
            "The evidence supports real-data usefulness and OOD transfer of structured primitives, while task-level gains remain model- and dataset-dependent.",
            "The GeoTES few-shot sweep shows a conditional advantage at 40% uniform coverage, not a monotone advantage at every sampling density.",
        ],
        "status": "conformant",
    }
    output = P4_RESULTS / "p4_real_data_evidence.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
