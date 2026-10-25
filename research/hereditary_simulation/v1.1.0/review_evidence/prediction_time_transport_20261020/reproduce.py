"""Run the extracted transport supplement without the author's workspace."""
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
        raise RuntimeError("Use the reproduce.py copy in the extracted transport supplement")
    subprocess.run([sys.executable, "-X", "utf8", str(snapshot/"experiments/run_prediction_time_transport.py")], check=True)
    sys.path.insert(0, str(snapshot/"experiments"))
    spec = importlib.util.spec_from_file_location("audit", folder/"experiments/audit_prediction_time_transport.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = snapshot
    module.OUT = snapshot/"results/prediction_time_transport_20261020/run_02"
    module.main()
    frozen = json.loads((folder/"result.json").read_text(encoding="utf-8"))
    fresh_path = module.OUT/"result.json"
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    assert frozen["summary"] == fresh["summary"]
    for a, b in zip(frozen["rows"], fresh["rows"]):
        for algorithm in a["algorithms"]:
            assert [r["state"] for r in a["algorithms"][algorithm]["rows"]] == [r["state"] for r in b["algorithms"][algorithm]["rows"]]
    receipt = {"all_checks_pass": True, "summaries_and_all_decisions_match": True,
        "frozen_result_sha256": hashlib.sha256((folder/"result.json").read_bytes()).hexdigest(),
        "fresh_result_sha256": hashlib.sha256(fresh_path.read_bytes()).hexdigest(),
        "scope": "Isolated numerical replay, not external mathematical review"}
    (folder/"isolated_replay_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
