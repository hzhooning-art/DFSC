"""Run from a fresh extracted prediction_time_dual_20261020 directory."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    destination = HERE/'isolated_replay'
    if destination.exists():
        raise FileExistsError('Use a fresh extraction for each replay')
    shutil.copytree(HERE/'source_snapshot',destination)
    parent = HERE.parent/'prediction_time_transport_20261020/result.json'
    target = destination/'results/prediction_time_transport_20261020/run_02/result.json'
    target.parent.mkdir(parents=True)
    shutil.copy2(parent,target)
    sys.path.insert(0,str(destination/'experiments'))
    runner = load('dual_replay',destination/'experiments/run_prediction_time_dual.py')
    runner.main()
    auditor = load('dual_audit_replay',destination/'experiments/audit_prediction_time_dual.py')
    auditor.main()
    fresh_dir = destination/'results/prediction_time_dual_20261020/run_01'
    original = json.loads((HERE/'result.json').read_text())
    fresh = json.loads((fresh_dir/'result.json').read_text())
    assert original['summary'] == fresh['summary']
    assert original['parent_portfolio_after_unresolved_only_patch'] == fresh['parent_portfolio_after_unresolved_only_patch']
    labels = lambda r:[(x['cohort'],x['scenario'],x['wait'],x['rho'],{k:v['state'] for k,v in x['algorithms'].items()}) for x in r['rows']]
    assert labels(original) == labels(fresh)
    audit = json.loads((fresh_dir/'audit.json').read_text())
    assert audit['all_checks_passed']
    receipt = {'all_checks_passed':True,'summaries_match':True,'all_candidate_labels_match':True,
        'fresh_lp_replays':audit['lp_replays'],'physical_80_digit_checks':audit['physical_witnesses_at_80_digits'],
        'frozen_result_sha256':hashlib.sha256((HERE/'result.json').read_bytes()).hexdigest(),
        'fresh_result_sha256':hashlib.sha256((fresh_dir/'result.json').read_bytes()).hexdigest()}
    (HERE/'isolated_replay_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
