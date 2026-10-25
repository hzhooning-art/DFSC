"""Unseen coordinate tests of unchanged scalar candidates, including refusal."""
import importlib
import time

import torch

from adapters import load_frozen_backends,connected_gradients
from records import exception_record
from references import scalar_reference,laplace_reference,finite


def assess(row,reference,p,within_domain):
    answer=dict(row,promotion_blocked=False)
    if not within_domain:
        answer['outcome']='outside_contract_'+row['status']
        answer['promotion_blocked']=row['status']!='refused'
    elif row['status']!='returned':
        answer['outcome']=row['status']
        answer['promotion_blocked']=row['status']=='error'
    elif reference['status']!='resolved':
        answer['outcome']='reference_unresolved'
    else:
        targets=[p['value_atol']+p['rtol']*abs(reference['values'][0])]+[
            p['gradient_atol']+p['rtol']*abs(v) for v in reference['values'][1:]]
        errors=[abs(v-r) for v,r in zip(row['values'],reference['values'])]
        good=all(e<=t for e,t in zip(errors,targets))
        answer.update(outcome='accurate_return' if good else 'inaccurate_return',errors=errors,targets=targets,
                      error_target_ratios=[e/t for e,t in zip(errors,targets)],promotion_blocked=not good)
    return answer


def useful_repair_gate(rows):
    return any(r.get('outcome')=='accurate_return' for r in rows) and not any(r.get('promotion_blocked') for r in rows)


def evaluate_stress(job,p):
    point=job['coordinate']; a,x=point['alpha'],point['radius']
    backend=load_frozen_backends(); package=backend['package']
    guard=importlib.import_module(package.__name__+'.guarded_series')
    result={'status':'completed','coordinate':point,'methods':[],'promotion_blocked':False}
    reference=(scalar_reference(a,x,p) if point['within_domain'] else {'status':'outside_candidate_contract'})
    result['reference']=reference
    if point['laplace_selected']:
        result['laplace']=laplace_reference(a,x,p)
        if result['laplace']['status']=='resolved' and reference['status']=='resolved':
            targets=[p['value_atol']+p['rtol']*abs(reference['values'][0])]+[
                p['gradient_atol']+p['rtol']*abs(v) for v in reference['values'][1:]]
            errors=[abs(v-r) for v,r in zip(result['laplace']['values'],reference['values'])]
            result['laplace'].update(errors_vs_defining=errors,targets=targets,agreement=all(e<=t for e,t in zip(errors,targets)))
            if not result['laplace']['agreement']:
                reference['status']='reference_unresolved'
    for method in ('guard','full_hp','original'):
        alpha=torch.tensor(a,dtype=torch.float64,requires_grad=True)
        z=torch.tensor(-x,dtype=torch.float64,requires_grad=True)
        start=time.perf_counter(); row={'method':method}
        try:
            if method=='guard':
                evaluation=guard.evaluate(alpha,z,rtol=p['rtol'],atol=p['value_atol'],gradient_atol=p['gradient_atol'])
                v=evaluation.values
                row['selected_terms']=int(evaluation.selected_terms)
            elif method=='full_hp':
                evaluation=package.evaluate_scalar(alpha,z)
                v=evaluation.values; row['diagnostics']=evaluation.diagnostics
            else:
                v=backend['original'].mittag_leffler_e(alpha,z,terms=180,method='hybrid')
            da,dz=connected_gradients(v,(alpha,z))
            row.update(status='returned',values=finite([float(v.detach()),float(da),float(dz)]).tolist())
        except Exception as exc:
            refused=isinstance(exc,(package.RepairRefusal,guard.NumericalRefusal)) or (
                isinstance(exc,ValueError) and not point['within_domain'])
            row.update(status='refused' if refused else 'error',**exception_record(exc))
        row['forward_backward_seconds']=time.perf_counter()-start
        row=assess(row,reference,p,point['within_domain'])
        result['methods'].append(row)
        if method=='full_hp' and row['promotion_blocked']:
            result['promotion_blocked']=True
    return result
