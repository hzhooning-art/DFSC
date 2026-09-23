from pathlib import Path
import hashlib
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RESULTS = ['uniform_crossover_results.json', 'local_limit_power_results.json', 'finite_local_limit_results.json', 'general_atom_results.json', 'phase_matched_score_results.json', 'cone_matched_score_results.json', 'score_power_curve_results.json', 'cone_direction_power_results.json', 'matched_glrt_results.json', 'score_design_stress_results.json', 'samp_direct_results.json', 'whitening_results.json', 'finite_training_calibration_results.json']
AUDITS = ['audit_uniform_crossover.py', 'audit_local_limit_power.py', 'audit_finite_local_limit.py', 'audit_general_atom.py', 'audit_phase_matched_score.py', 'audit_score_innovations.py', 'audit_matched_glrt.py', 'audit_score_design_stress.py', 'audit_whitening.py', 'audit_finite_training_calibration.py']

for name in RESULTS:
    data = json.loads((HERE / name).read_text(encoding='utf-8'))
    for filename, expected in data.get('hashes', {}).items():
        actual = hashlib.sha256((HERE / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f'hash mismatch: {filename}')

# Verify the public SAMP record without redistributing the third-party tree.
samp = json.loads((HERE / 'samp_direct_results.json').read_text(encoding='utf-8'))
assert all(samp['gates'].values())
assert samp['source']['commit'] == '3f8cf58a32d7c7849da89ef911827fbfd57a11bd'
assert samp['official_SAMP']['null']['applicable'] is False
assert samp['official_SAMP']['two_rate']['applicable'] is False

for name in AUDITS:
    subprocess.run([sys.executable, name], cwd=HERE, check=True)

print(f'PASS: {len(RESULTS)} frozen results and {len(AUDITS)} audits')
