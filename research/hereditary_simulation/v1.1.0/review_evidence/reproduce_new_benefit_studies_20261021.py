"""Replay from an extracted review supplement into a NEW directory.

Usage: python reproduce_new_benefit_studies_20261021.py SUPPLEMENT_ROOT NEW_PARENT
Run with Python including numpy, scipy, mpmath and openpyxl.
"""
from pathlib import Path
import argparse,shutil,subprocess,sys,json
def main():
    p=argparse.ArgumentParser();p.add_argument('supplement',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
    base=a.supplement.resolve();target=a.destination.resolve()/'P4'
    if target.exists():raise FileExistsError('Replay destination must be new')
    for name in ('matched_method_benefit_20261021','orbital_hold_pilot_20261021'):
        snapshot=base/name/'source_snapshot'
        for src in snapshot.rglob('*'):
            if not src.is_file() or src.name=='extracted.json':continue
            dest=target/src.relative_to(snapshot);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    parent=base/'prediction_time_transport_20261020/result.json'
    assert parent.exists(),parent
    dest=target/'results/prediction_time_transport_20261020/run_02/result.json';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent,dest)
    shutil.copytree(base/'orbital_hold_pilot_20261021/source_data',target/'data/external/orbital_fat_20261021')
    for command in ('tools/extract_orbital_hold_20261021.py','experiments/run_matched_method_benefit_20261021.py',
                    'experiments/audit_matched_method_benefit_20261021.py','experiments/run_orbital_hold_pilot_20261021.py',
                    'experiments/audit_orbital_hold_pilot_20261021.py'):
        subprocess.run([sys.executable,str(target/command)],check=True,cwd=target)
    material=json.loads((target/'results/orbital_hold_pilot_20261021/audit.json').read_text())
    assert material['all_numerical_checks_passed'] and not material['physical_validation_passed']
    print('Replay finished. The failed physical qualification is an expected retained result.')
if __name__=='__main__':main()
