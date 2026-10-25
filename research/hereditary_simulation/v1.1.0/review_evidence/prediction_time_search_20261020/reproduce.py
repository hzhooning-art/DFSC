"""Run from the extracted prediction_time_search_20261020 supplement folder.

Requires numpy, scipy and mpmath. Generates fresh results inside source_snapshot
and compares decisions/work counts with the frozen result, without overwriting
the supplied result or relying on the author's workspace.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys


def main():
    folder = Path(__file__).resolve().parent
    snapshot = folder/"source_snapshot"
    if not snapshot.exists():
        raise RuntimeError("Run the copy named reproduce.py in the extracted supplement folder")
    subprocess.run([sys.executable, "-X", "utf8", str(snapshot/"experiments/run_prediction_time_search.py")], check=True)
    sys.path.insert(0, str(snapshot/"experiments"))
    module_spec = importlib.util.spec_from_file_location("replay_audit", folder/"experiments/audit_prediction_time_search.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.ROOT = snapshot
    module.OUT = snapshot/"results/prediction_time_search_20261020/run_05"
    module.main()
    frozen = json.loads((folder/"result.json").read_text(encoding="utf-8"))
    fresh_path = module.OUT/"result.json"
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    assert frozen["summary"] == fresh["summary"]
    assert not fresh["opposite_decisive_labels"]
    for a, b in zip(frozen["rows"], fresh["rows"]):
        for algorithm in a["algorithms"]:
            assert [r["state"] for r in a["algorithms"][algorithm]["rows"]] == [r["state"] for r in b["algorithms"][algorithm]["rows"]]
    receipt = {"all_checks_pass": True, "summary_and_all_decisions_match": True,
               "frozen_result_sha256": hashlib.sha256((folder/"result.json").read_bytes()).hexdigest(),
               "fresh_result_sha256": hashlib.sha256(fresh_path.read_bytes()).hexdigest(),
               "scope": "Numerical replay, not external mathematical review or directed-rounding verification"}
    (folder/"isolated_replay_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
