"""Accuracy-matched costs and sensitivity/inverse audits on the common problem."""
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import adapters as a
from records import write_checkpoint,exception_record
from references import action_reference, independent_action, finite


def initial_alpha(seed):
    return float(np.random.Generator(np.random.PCG64(seed)).uniform(.6,.8))


def qualifies(row,band):
    return (row.get('status')=='returned' and row.get('reference_status')=='resolved'
            and row['relative_l2']<=band and row['max_abs']<=1e-6+band*row['max_abs_reference'])


def timing_summary(samples):
    return {'median_seconds':statistics.median(samples),'samples':samples}


def errors(values,reference):
    v,r=finite(values),finite(reference)
    return {'relative_l2':float(np.linalg.norm(v-r)/np.linalg.norm(r)),
            'max_abs':float(np.max(np.abs(v-r))),'max_abs_reference':float(np.max(np.abs(r)))}


def train_alpha(job,p,checkpoint_path):
    problem=a.problem('rotated2',job['truth_alpha'])
    reference=action_reference(problem['operator'],problem['u0'],p['a_times'],job['truth_alpha'],p)
    test_reference=action_reference(problem['operator'],problem['u0'],p['a_test_times'],job['truth_alpha'],p)
    result={'status':'reference_unresolved','reference':reference,'test_reference':test_reference,
            'trajectory':[],'completed_updates':0,'initial_alpha':initial_alpha(job['seed']),
            'gradient_label':'explicit_centered_FD' if job['method']=='pycaputo' else 'native_torch_AD'}
    if reference['status']!='resolved' or test_reference['status']!='resolved':
        return result
    target=torch.tensor(reference['values'],dtype=torch.float64)
    start=time.perf_counter()
    alpha=torch.nn.Parameter(torch.tensor(result['initial_alpha'],dtype=torch.float64))
    optimizer=torch.optim.Adam([alpha],lr=p['a_inverse_lr'])
    method=job['method']; deadline=p.get('deadline_monotonic',math.inf)
    result['status']='partial'
    # Warmup is paid by parent clock, excluded from the primary training time.
    warm_start=time.perf_counter()
    result['setup_seconds']=warm_start-start
    warm=a.forward(method,problem['operator'],problem['u0'],p['a_times'],alpha,job['steps'],method!='pycaputo')
    if warm['status']!='returned':
        result.update(status=warm['status'],warmup={k:v for k,v in warm.items() if k!='values'})
        return result
    result['warmup_seconds']=time.perf_counter()-warm_start
    train_start=time.perf_counter()
    for i in range(p['a_inverse_steps']):
        if time.perf_counter()+.1>=deadline:
            break
        before=float(alpha.detach()); update_start=time.perf_counter(); optimizer.zero_grad()
        row=a.forward(method,problem['operator'],problem['u0'],p['a_times'],alpha,job['steps'],method!='pycaputo')
        if row['status']!='returned':
            result.update(status=row['status'],failure={k:v for k,v in row.items() if k!='values'}); break
        loss=torch.mean((row['values']-target)**2)
        try:
            if method=='pycaputo':
                h=p['a_inverse_fd_step']; losses=[]
                for offset in (-h,h):
                    other=a.forward(method,problem['operator'],problem['u0'],p['a_times'],before+offset,job['steps'],False)
                    if other['status']!='returned':
                        raise RuntimeError('FD solve failed: '+str(other))
                    losses.append(float(torch.mean((other['values']-target)**2)))
                grad=(losses[1]-losses[0])/(2*h)
                alpha.grad=torch.tensor(grad,dtype=torch.float64)
            else:
                grad_tensor=a.connected_gradients(loss,(alpha,))[0]
                alpha.grad=grad_tensor; grad=float(grad_tensor)
            if not math.isfinite(grad):
                raise ArithmeticError('Nonfinite training gradient; no update')
            optimizer.step()
            with torch.no_grad(): alpha.clamp_(.45,.95)
            after=float(alpha.detach())
            if not math.isfinite(after):
                raise ArithmeticError('Nonfinite optimizer output')
        except Exception as exc:
            result.update(status='error',**exception_record(exc)); break
        result['trajectory'].append({'step':i+1,'alpha_before':before,'alpha_after':after,
            'loss':float(loss.detach()),'gradient':grad,'update_seconds':time.perf_counter()-update_start})
        result['completed_updates']=i+1
        write_checkpoint(checkpoint_path,dict(status='partial',completed_updates=i+1,
                                             final_alpha=after,trajectory=result['trajectory']))
    result['final_alpha']=float(alpha.detach())
    result['training_seconds']=time.perf_counter()-train_start
    if result['completed_updates']==p['a_inverse_steps']:
        result['status']='completed'
    if time.perf_counter()+.1<deadline:
        final=a.forward(method,problem['operator'],problem['u0'],p['a_test_times'],alpha,job['steps'],False)
        if final['status']=='returned':
            result['prediction']=errors(final['values'].detach().numpy(),test_reference['values'])
    result['is_sixty_step_run']=result['completed_updates']==60
    result['recovery']=(result['status']=='completed' and result['is_sixty_step_run'] and
                        abs(result['final_alpha']-job['truth_alpha'])<=p['parameter_recovery_abs'] and
                        result.get('prediction',{}).get('relative_l2',math.inf)<=p['prediction_recovery_l2'])
    result['total_seconds_excluding_reference_imports_and_warmup']=result['setup_seconds']+time.perf_counter()-train_start
    return result


def evaluate_external(job,p):
    if job['kind']=='preflight':
        cases=[]
        for method in ('selective','full_hp','original','fdeint','pycaputo'):
            alpha=torch.nn.Parameter(torch.tensor(.73,dtype=torch.float64))
            row=a.forward(method,[[1.]],[[.8]],[0.,.17,.41],alpha,
                          8 if method in ('fdeint','pycaputo') else None,method!='pycaputo')
            cases.append(dict(method=method,**{k:(v.detach().tolist() if torch.is_tensor(v) else v) for k,v in row.items()}))
        return {'status':'completed','excluded_from_scientific_counts':True,'forward_cases':cases}
    if job['kind']=='inverse':
        return train_alpha(job,p,Path(p['checkpoint_path']))
    problem=a.problem(job['problem'],job['alpha'])
    reference=action_reference(problem['operator'],problem['u0'],problem['times'],job['alpha'],p)
    result={'status':'reference_unresolved','reference_status':reference['status'],'reference':reference}
    if reference['status']!='resolved':
        return result
    if job['kind']=='forward':
        calls=[]
        for i in range(1+p['timed_repeats']):
            row=a.forward(job['method'],problem['operator'],problem['u0'],problem['times'],job['alpha'],job['steps'],False)
            raw={k:v for k,v in row.items() if k!='values'}
            raw['warmup']=i==0
            if row['status']=='returned':
                raw['values']=row['values'].detach().tolist()
            calls.append(raw)
            if row['status']!='returned':
                return dict(result,status=row['status'],calls=calls)
        result.update(status='returned',calls=calls,**errors(calls[-1]['values'],reference['values']),
                      timing=timing_summary([c['elapsed_seconds'] for c in calls[1:]]))
        result['qualified_bands']={str(b):qualifies(result,b) for b in p['accuracy_bands']}
    else:
        start=time.perf_counter()
        gradient=a.loss_gradient(job['method'],problem,job['alpha_path'],job['steps'])
        result.update(status=gradient['status'],gradient=gradient,total_gradient_audit_seconds=time.perf_counter()-start)
        if gradient['status']=='returned':
            reference_a=reference['alpha_loss_derivative']['value']
            target=p['gradient_target_atol']+p['gradient_target_rtol']*abs(reference_a)
            ea=abs(gradient['alpha_gradient']-reference_a)
            eu=np.abs(np.asarray(gradient['u0_gradient'])-np.asarray(reference['u0_gradient']))
            tu=p['gradient_target_atol']+p['gradient_target_rtol']*np.abs(reference['u0_gradient'])
            result['continuous_gradient_check']={'alpha_error':ea,'alpha_target':target,
                'u0_errors':eu.tolist(),'u0_targets':tu.tolist(),'all_pass':bool(ea<=target and (eu<=tu).all())}
    return result
