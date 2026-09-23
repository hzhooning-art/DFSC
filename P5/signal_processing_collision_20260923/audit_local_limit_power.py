import json
import math
from pathlib import Path

from local_limit_power import build, power


HERE = Path(__file__).resolve().parent
stored = json.loads((HERE / "local_limit_power_results.json").read_text(encoding="utf-8"))
fresh = build()
assert stored["hashes"] == fresh["hashes"]
assert stored["profile_residual_df"] == 131
assert stored["finite_union_reference_df"] == 132

for key in (
    "profile_critical",
    "finite_union_asymptotic_critical",
    "rho_for_target_profile",
    "rho_for_target_exact_union_local_limit",
):
    assert math.isclose(stored[key], fresh[key], rel_tol=1e-13)

assert stored["profile_critical"] < stored["finite_union_asymptotic_critical"]
assert stored["rho_for_target_profile"] < stored["rho_for_target_exact_union_local_limit"]
assert math.isclose(
    power(
        stored["rho_for_target_profile"],
        stored["profile_residual_df"],
        stored["profile_critical"],
    ),
    stored["target_power"],
    abs_tol=1e-12,
)
assert math.isclose(
    power(
        stored["rho_for_target_exact_union_local_limit"],
        stored["profile_residual_df"],
        stored["finite_union_asymptotic_critical"],
    ),
    stored["target_power"],
    abs_tol=1e-12,
)

for old, new in zip(stored["rows"], fresh["rows"]):
    assert old.keys() == new.keys()
    for key in old:
        assert math.isclose(old[key], new[key], rel_tol=1e-13, abs_tol=1e-14)

print("local limit power audit passed")
