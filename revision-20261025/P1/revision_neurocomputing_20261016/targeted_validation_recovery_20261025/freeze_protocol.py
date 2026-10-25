"""Generate all scientific settings and identities before inspecting outcomes."""
import importlib.metadata
import itertools
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from records import read, sha, write_terminal

HERE=Path(__file__).resolve().parent
REV=HERE.parent
WHEEL_SHA='467aa4de1fed0b9de7aed53d33cccf8455a874f32e442c85f5e8593ff8b415fb'
METHODS=['selective','full_hp','original','fdeint','pycaputo']


def build_protocol():
    return {'version':'1.0.0', 'author_approved':'2026-10-24', 'seed':20261023,
        'post_existing_results_not_blind_preregistration':True,
        'cpu_threads':1, 'dtype':'float64', 'track_budget_seconds':1800,
        'worker_limit_seconds':120, 'training_limit_seconds':180,
        'reference_digits':[300,380], 'reference_max_terms':6000,
        'reference_pair_rtol':1e-35, 'reference_tiny_terms':8,
        'fd_steps':[.01,.005,.0025], 'laplace_digits':[55,85],
        'independent_value_atol':1e-11, 'independent_value_rtol':1e-9,
        'independent_gradient_atol':2e-8, 'independent_gradient_rtol':2e-6,
        'value_atol':1e-12, 'gradient_atol':1e-10, 'rtol':1e-9,
        'gradient_target_atol':1e-7, 'gradient_target_rtol':1e-3,
        'accuracy_bands':[1e-3,1e-4], 'a_times':[0,.125,.25,.5,.75,1],
        'a_test_times':[.0625,.1875,.375,.625,.875,1.25],
        'a_alphas':[.5,.7,.9], 'external_steps':[64,128,256,512,1024],
        'gradient_steps':[64,256,1024], 'warmups':1, 'timed_repeats':3,
        'a_inverse_truth':[.55,.85], 'seeds':[101,211,307],
        'a_inverse_steps':60, 'a_inverse_lr':.02, 'a_inverse_resolution':256,
        'a_inverse_fd_step':1e-4, 'c_sizes':[4,8,16],
        'c_times':[.1,.3,.6,1], 'c_test_times':[.15,.45,.8,1.5],
        'c_updates':100, 'c_lr':.05, 'c_warmups':1,
        'paired_steps':[1,12,25,50,100], 'pair_atol':1e-7, 'pair_rtol':1e-6,
        'parameter_recovery_abs':.02, 'prediction_recovery_l2':1e-2,
        'optional_wheel_sha256':WHEEL_SHA,
        'execution':'preserved_source_snapshot_verified_against_wheel_not_fresh_install',
        'a_structural_advantage':'Known spectral propagator versus history-aware time steppers',
        'timing_exclusions':['imports','process_startup','reference_generation'],
        'parent_budget_includes_exclusions':True,
        'schedule_order':'A preflight then interleaved forward/gradient/inverse; B rotated uniform/log/boundary; C seed outer then six operators, rotated methods',
        'no_candidate_tuning':True, 'scientific_global_certificate':False}


def build_coordinates(previous):
    rng=np.random.Generator(np.random.PCG64(20261023))
    seen=set(previous)
    rows=[]
    for stratum in ('uniform','log_uniform'):
        for i in range(144):
            while True:
                a=float(rng.uniform(.45,1.95))
                x=float(rng.uniform(0,16) if stratum=='uniform' else
                        np.exp(rng.uniform(np.log(1e-6),np.log(16))))
                if (a,x) not in seen:
                    break
            seen.add((a,x))
            rows.append({'id':f'{stratum}-{i:03}', 'stratum':stratum,'alpha':a,'radius':x})
    boundary=[]
    for a0,e,x in itertools.product([.6,1.],[-1e-6,1e-6],[3.4,4.6,6.8,9.2,10.2,13.8]):
        boundary.append(('branch',a0+e,x))
    for a0,e,x in itertools.product([.45,1.95],[-1e-8,0,1e-8],[.2,12.]):
        boundary.append(('alpha_edge',a0+e,x))
    for x0,e,a in itertools.product([0.,16.],[-1e-8,0,1e-8],[.55,1.5]):
        boundary.append(('radius_edge',a,x0+e))
    for i,(kind,a,x) in enumerate(boundary):
        rows.append({'id':f'boundary-{i:02}', 'stratum':kind, 'alpha':a,'radius':x})
    laplace=set()
    for stratum in ('uniform','log_uniform'):
        group=sorted([r for r in rows if r['stratum']==stratum],key=lambda r:(r['alpha'],r['radius'],r['id']))
        laplace.update(group[round(j*143/7)]['id'] for j in range(8))
    appeared=set()
    for r in rows:
        coord=(r['alpha'],r['radius'])
        r.update(within_domain=.45<=coord[0]<=1.95 and 0<=coord[1]<=16,
                 overlaps_previous=coord in previous, duplicates_earlier=coord in appeared,
                 laplace_selected=r['id'] in laplace)
        appeared.add(coord)
    return rows


def build_schedule(p, coordinates):
    forward=[]; gradient=[]; inverse=[]
    configs=list(itertools.product(['scalar','rotated2'],p['a_alphas']))
    # Resolution round then problem then rotated method; direct methods only once.
    for ri,steps in enumerate(p['external_steps']):
        for ci,(problem,a) in enumerate(configs):
            order=METHODS[ci%5:]+METHODS[:ci%5]
            for method in order:
                if ri and method not in ('fdeint','pycaputo'):
                    continue
                forward.append(dict(kind='forward',problem=problem,alpha=a,method=method,
                                    steps=steps if method in ('fdeint','pycaputo') else None))
    for ri,steps in enumerate(p['gradient_steps']):
        for ci,(problem,a) in enumerate(configs):
            order=METHODS[ci%5:]+METHODS[:ci%5]
            for method in order:
                if ri and method not in ('fdeint','pycaputo'):
                    continue
                paths=['physical','transformed'] if method=='fdeint' else ['physical']
                for path in paths:
                    gradient.append(dict(kind='gradient',problem=problem,alpha=a,method=method,
                        alpha_path=path,steps=steps if method in ('fdeint','pycaputo') else None))
    for si,(a,seed) in enumerate(itertools.product(p['a_inverse_truth'],p['seeds'])):
        order=['selective','full_hp','fdeint','pycaputo']
        order=order[si%4:]+order[:si%4]
        for method in order:
            inverse.append(dict(kind='inverse',problem='rotated2',truth_alpha=a,seed=seed,method=method,
                                steps=256 if method in ('fdeint','pycaputo') else None))
    jobs=[dict(track='A',kind='preflight',id='A-preflight')]
    for triple in itertools.zip_longest(forward,gradient,inverse):
        for row in triple:
            if row is not None:
                jobs.append(dict(row,track='A',id=f'A-{len(jobs):04}'))
    strata=[[r for r in coordinates if r['stratum']==name] for name in ('uniform','log_uniform')]
    strata.append([r for r in coordinates if r['stratum'] not in ('uniform','log_uniform')])
    for triple in itertools.zip_longest(*strata):
        for r in triple:
            if r is not None:
                jobs.append(dict(track='B',kind='stress',coordinate=r,id='B-'+r['id']))
    for si,(seed,size,regime) in enumerate(itertools.product(p['seeds'],p['c_sizes'],['mild','hard'])):
        order=['guard','selective','full_hp','original']
        order=order[si%4:]+order[:si%4]
        for method in order:
            jobs.append(dict(track='C',kind='joint',id=f'C-{size}-{regime}-{seed}-{method}',
                             size=size,regime=regime,seed=seed,method=method))
    return jobs


def default_dependencies():
    repo=REV.parents[1]
    deps={str(p.relative_to(HERE)):p for p in HERE.glob('*.py')}
    deps.update({'test/'+p.name:p for p in (HERE/'tests').glob('test_*.py')})
    for rel in ['repair_feasibility/results/actual_repair_feasibility.json',
                'software_snapshot/dfsc/mittag_leffler.py',
                'cpu_repair_delivery_20261021/output/dist/dfsc_cpu_repair_candidate-0.1.0.dev0-py3-none-any.whl',
                ]:
        deps['prior/'+rel]=REV/rel
    for path in (REV/'cpu_repair_delivery_20261021/backend/src/dfsc_cpu_repair').glob('*.py'):
        deps['backend/'+path.name]=path
    deps['design']=repo/'docs/superpowers/specs/2026-10-23-p1-targeted-validation-design.md'
    deps['plan']=repo/'docs/superpowers/plans/2026-10-24-p1-targeted-validation.md'
    return deps


def freeze(root=HERE, *, dependencies=None, previous=None):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    if any((root/n).exists() for n in ('protocol.json','coordinates.json','schedule.json','freeze_receipt.json')):
        raise FileExistsError('Existing freeze is immutable; new version required')
    dependencies=default_dependencies() if dependencies is None else dependencies
    if previous is None:
        old=read(REV/'repair_feasibility/results/actual_repair_feasibility.json')
        previous={(r['alpha'],r['radius']) for r in old['rows']}
        if len(previous)!=217:
            raise ValueError('Previous coordinate identity mismatch')
    p=build_protocol(); coords=build_coordinates(previous); jobs=build_schedule(p,coords)
    bindings={name:{'path':os.path.relpath(Path(path).resolve(),root.resolve()),'sha256':sha(path)} for name,path in dependencies.items()}
    for name,payload in [('protocol.json',p),('coordinates.json',coords),('schedule.json',jobs)]:
        write_terminal(root/name,payload)
        bindings[name]={'path':name,'sha256':sha(root/name)}
    versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','mpmath','pymittagleffler','FDEint','pycaputo')}
    try:
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        git_commit=None
    receipt={'bindings':bindings,'versions':versions,'python':sys.version,'platform':platform.platform(),
             'git_commit':git_commit,'git_provenance_status':'available' if git_commit else 'unavailable',
             'source_snapshot_execution':True,'previous_points':len(previous),
             'science_started':False,'clock_basis':'parent active wall time including child startup/references'}
    if root.resolve()==HERE:
        receipt['preserved_artifacts']={rel:sha(REV/rel) for rel in (
            'revise_delivery_20261022/output/pdf/Manuscript_Clean.pdf',
            'cpu_repair_delivery_20261021/manuscript/manuscript.tex',
            'output/DFSC_Revise_Author_Review_Delivery_20261022.zip')}
    receipt['installed_source_hashes']={}
    for module in ('FDEint.FixedStepFDEint','pycaputo.fode.caputo','pycaputo.controller','pycaputo.stepping'):
        spec=__import__('importlib.util',fromlist=['find_spec']).find_spec(module)
        receipt['installed_source_hashes'][module]={'path':spec.origin,'sha256':sha(spec.origin)}
    write_terminal(root/'freeze_receipt.json',receipt)
    return receipt


if __name__=='__main__':
    print(freeze())
