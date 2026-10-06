import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", choices=("pass", "fail"), default="pass")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--invalid-json", action="store_true")
    args = parser.parse_args()
    if args.sleep:
        time.sleep(args.sleep)
    if args.invalid_json:
        print("not-json")
        return
    print(
        json.dumps(
            {
                "test_id": "reference.value",
                "outcome": args.outcome,
                "invocations": 1,
                "details": {"runner": "fake"},
            }
        )
    )


if __name__ == "__main__":
    main()
