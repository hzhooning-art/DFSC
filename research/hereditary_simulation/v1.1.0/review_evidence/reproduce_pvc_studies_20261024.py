"""Replay both supplied PVC studies into a NEW destination, using only this supplement.

Usage: python reproduce_pvc_studies_20261024.py SUPPLEMENT_ROOT NEW_DESTINATION
Requires Python, numpy, scipy, mpmath and openpyxl. No network needed.
"""
from pathlib import Path
import hashlib, json, math, os, shutil, subprocess, sys
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable(v):
    if isinstance(v,dict): return {k:stable(x) for k,x in v.items() if k not in ('total_seconds','feasibility_seconds','elapsed_seconds','lp_witnesses_sha256')}
    if isinstance(v,list): return [stable(x) for x in v]
    return v
def compare(a,b,path=''):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a: compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+'/'+str(i))
    elif a != b:
        assert isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(path,a,b)

def main():
    supplied=Path(sys.argv[1]).resolve();dest=Path(sys.argv[2]).resolve()
    if dest.exists(): raise FileExistsError('Use a new destination; no results are overwritten')
    folders=('polymer_glass_hold_20261021','pvc_time_search_benefit_20261023')
    manifest=json.loads((supplied/'manifest.json').read_text())['files']
    # Check every supplied entry, not only the files selected for this replay.
    for name,item in manifest.items():
        path=supplied/name
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
    source={}
    for folder in folders:
        for p in (supplied/folder/'source_snapshot').rglob('*'):
            if p.is_file():
                rel=p.relative_to(supplied/folder/'source_snapshot')
                if rel.parts[0]=='results': continue
                if rel in source: assert sha(source[rel])==sha(p),str(rel)
                source[rel]=p
    dest.mkdir(parents=True)
    for rel,p in source.items():
        q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    spec=json.loads((dest/'experiments/specs/polymer_glass_hold_20261021.json').read_text())
    raw=supplied/'polymer_glass_hold_20261021/source_data'/spec['workbook']
    assert sha(raw)==spec['workbook_sha256']
    q=dest/'data/external/polymer_glass_20261021'/spec['workbook'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(raw,q)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): env[key]='1'
    for relative in ('tools/extract_polymer_glass_hold_20261021.py',
                     'experiments/run_polymer_glass_hold_20261021.py',
                     'experiments/audit_polymer_glass_hold_20261021.py',
                     'experiments/run_pvc_time_search_benefit_20261023.py',
                     'experiments/audit_pvc_time_search_benefit_20261023.py'):
        subprocess.run([sys.executable,str(dest/relative)],cwd=dest,env=env,check=True)
    checks={};arrays=0
    for folder in folders:
        original=supplied/folder;fresh=dest/'results'/folder
        a=json.loads((original/'result.json').read_text());b=json.loads((fresh/'result.json').read_text())
        if folder==folders[0]:
            compare(stable(a),stable(b))
            assert sha(original/'extracted.json')==sha(fresh/'extracted.json')
            flag='all_numerical_checks_passed'
        else:
            compare(stable(a['timing_rows']),stable(b['timing_rows']))
            for key in ('cases','correlated_requests','spec_sha256','protocol_sha256','by_case','budget_results','opposite_decisive_labels','all_label_differences','accepted_against_raw_targets','claimed_physical_guarantee'): compare(a[key],b[key])
            assert sha(original/'cases.json')==sha(fresh/'cases.json')
            for path in original.glob('T*.json'): assert stable(json.loads(path.read_text()))==stable(json.loads((fresh/path.name).read_text()))
            flag='all_checks_passed'
        for p in original.glob('*.npz'):
            with np.load(p) as x,np.load(fresh/p.name) as y:
                assert x.files==y.files
                for key in x.files:
                    assert np.array_equal(x[key],y[key]),(folder,p.name,key)
                    arrays+=1
        assert json.loads((fresh/'audit.json').read_text())[flag]
        checks[folder]={'numerical_outcomes_match':True,'all_witness_arrays_identical':True,'full_audit_passed':True,'original_result_sha256':sha(original/'result.json'),'replay_result_sha256':sha(fresh/'result.json')}
        if folder==folders[1]: checks[folder]['replayed_median_speed_ratio']=b['median_speed_ratio']
    receipt={'all_checks_passed':True,'studies':checks,'arrays_checked':arrays,'timing_variation_allowed':True,'calibrated_material_guarantee_established':False}
    (dest/'replay_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
