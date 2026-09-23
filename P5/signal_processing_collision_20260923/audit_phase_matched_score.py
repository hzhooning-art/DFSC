import hashlib
import json

from phase_matched_score import HERE, MS, one_trial, population_row


def main():
    result = json.loads((HERE / "phase_matched_score_results.json").read_text(encoding="utf-8"))
    assert len(result["rows"]) == 16
    for name, expected in result["hashes"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected
    for row in result["rows"]:
        pop = population_row(row["path"], row["m"])
        for key, value in pop.items():
            assert abs(value - row[key]) <= 1e-10 * max(1, abs(value))
        for replay in row["replay"]:
            assert one_trial(row["path"], row["m"], replay["seed"]) == replay
    assert result["settings"]["m_values"] == list(MS)
    assert all(result["gates"].values())
    print("phase-matched score audit passed")


if __name__ == "__main__":
    main()
