"""Explicit fault controls for observation handling; no numerical defect claims."""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--test-id", required=True)
parser.add_argument("--mode", choices=["invalid_json", "wrong_id", "zero_calls", "non_evaluable"], required=True)
args = parser.parse_args()
if args.mode == "invalid_json":
    print("not JSON")
else:
    print(json.dumps({"schema": "P4-Relation-Observation-v1",
                      "test_id": "reference.other" if args.mode == "wrong_id" else args.test_id,
                      "outcome": "NON_EVALUABLE" if args.mode == "non_evaluable" else "PASS",
                      "subject_invocations": 0 if args.mode in {"zero_calls", "non_evaluable"} else 2,
                      "details": {"controlled_mode": args.mode}}))
