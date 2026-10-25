"""Matched larger SPD inversions, explicit refusal and completed-step checkpoints."""
import math
import time

import numpy as np
import torch

import adapters as a
from external_comparison import errors
from records import write_checkpoint,exception_record
from references import action_reference,finite


def scenario(size,regime,seed):
    ri={'mild':0,'hard':1}[regime]
    basis_seed=20261024+10*size+ri
    rng=np.random.Generator(np.random.PCG64(basis_seed))
    q,r=np.linalg.qr(rng.normal(size=(size,size)))
    signs=np.where(np.diag(r)>=0,1.,-1.)
    q=q*signs[None,:]
    lam=np.linspace(.2,1.4 if regime=='mild' else 6.,size)
    operator=q@np.diag(lam)@q.T
    u0=rng.normal(size=(2,size))
    raw=np.random.Generator(np.random.PCG64(seed)).normal([.4,-.1],.1,2)
    maximum_radius=1.5**.95*max(lam)  # lambda_max>1 and beta<=2
    if maximum_radius>=16:
        raise ValueError('Frozen parameter box leaves repaired domain')
    return {'size':size,'regime':regime,'basis_seed':basis_seed,
            'qr_sign_convention':'positive R diagonal; Q columns multiplied by sign(diag(R)), zero→+1',
            'operator':operator.tolist(),'eigenvalues':lam.tolist(),'u0':u0.tolist(),
            'raw_initial':raw.tolist(),'truth':[.72,1.4] if regime=='mild' else [.55,1.8],
            'maximum_box_radius':float(maximum_radius)}


def physical(raw):
    return torch.stack([.45+.5*torch.sigmoid(raw[0]),1.+torch.sigmoid(raw[1])])


def train_joint(job,p,checkpoint_path):
    case=scenario(job['size'],job['regime'],job['seed'])
    result={'status':'reference_unresolved','scenario':case,'trajectory':[],
            'raw_initial':case['raw_initial'],'raw_final':case['raw_initial'],
            'completed_updates':0,'is_hundred_step_run':False,'recovery':False,'promotion_blocked':False}
    ref_settings=dict(p,beta=case['truth'][1])
    train_ref=action_reference(case['operator'],case['u0'],p['c_times'],case['truth'][0],ref_settings)
    test_ref=action_reference(case['operator'],case['u0'],p['c_test_times'],case['truth'][0],ref_settings)
    result.update(train_reference=train_ref,test_reference=test_ref)
    if train_ref['status']!='resolved' or test_ref['status']!='resolved':
        return result
    a.load_frozen_backends()  # Explicitly excluded import/source-verification setup.
    begin=time.perf_counter(); deadline=p.get('deadline_monotonic',math.inf)
    model=a.build_action(job['method'],case['operator'])
    u0=torch.tensor(case['u0'],dtype=torch.float64)
    times=torch.tensor(p['c_times'],dtype=torch.float64)
    target=torch.tensor(train_ref['values'],dtype=torch.float64)
    raw=torch.nn.Parameter(torch.tensor(case['raw_initial'],dtype=torch.float64))
    optimizer=torch.optim.Adam([raw],lr=p['c_lr'])
    package=a.load_frozen_backends()['package']
    result['setup_seconds']=time.perf_counter()-begin
    result['status']='partial'
    try:
        warm_start=time.perf_counter()
        aa,bb=physical(raw)
        warm,_=a.action_values(model(u0,times,aa,beta=bb))
        a.connected_gradients(torch.mean((warm-target)**2),(raw,))
        result['warmup_seconds']=time.perf_counter()-warm_start
        train_start=time.perf_counter()
        for step in range(p['c_updates']):
            if time.perf_counter()+.1>=deadline:
                break
            step_start=time.perf_counter()
            raw_before=raw.detach().tolist(); before=physical(raw).detach().tolist()
            optimizer.zero_grad()
            aa,bb=physical(raw)
            values,counts=a.action_values(model(u0,times,aa,beta=bb))
            loss=torch.mean((values-target)**2)
            gradient=a.connected_gradients(loss,(raw,))[0]
            finite(gradient.detach().numpy())
            raw.grad=gradient
            optimizer.step()
            finite(raw.detach().numpy())
            after=physical(raw).detach().tolist()
            row={'step':step+1,'raw_before':raw_before,'raw_after':raw.detach().tolist(),
                 'physical_before':before,'physical_after':after,'raw_gradient':gradient.detach().tolist(),
                 'loss':float(loss.detach()),'update_seconds':time.perf_counter()-step_start,
                 'training_cumulative_seconds':time.perf_counter()-train_start,**counts}
            result['trajectory'].append(row)
            result.update(completed_updates=step+1,raw_final=row['raw_after'])
            write_checkpoint(checkpoint_path,{'status':'partial','stage':'completed_update',
                'completed_updates':step+1,'raw_final':row['raw_after'],'trajectory':result['trajectory'],
                'scenario':case,'model_method':job['method']})
        result['training_seconds']=time.perf_counter()-train_start
        if result['completed_updates']==p['c_updates']:
            result['status']='completed'
        if time.perf_counter()+.1<deadline:
            final_start=time.perf_counter()
            with torch.no_grad():
                aa,bb=physical(raw)
                output,counts=a.action_values(model(u0,torch.tensor(p['c_test_times'],dtype=torch.float64),aa,beta=bb))
                result['prediction']=errors(output.numpy(),test_ref['values'])
                train_output,_=a.action_values(model(u0,times,aa,beta=bb))
                result['final_training_loss']=float(torch.mean((train_output-target)**2))
            result['final_evaluation_seconds']=time.perf_counter()-final_start
    except (package.ActionRefusal,package.RepairRefusal) as exc:
        result.update(status='refused',**exception_record(exc))
    except Exception as exc:
        result.update(status='error',promotion_blocked=job['method'] in ('selective','full_hp'),**exception_record(exc))
    result['raw_final']=raw.detach().tolist()
    result['physical_final']=physical(raw).detach().tolist()
    errs=[abs(v-t) for v,t in zip(result['physical_final'],case['truth'])]
    result['parameter_absolute_errors']=errs
    result['parameter_normalized_errors']=[errs[0]/.5,errs[1]/1.]
    result['is_hundred_step_run']=result['completed_updates']==100
    result['recovery']=(result['status']=='completed' and result['is_hundred_step_run'] and
                       max(errs)<=p['parameter_recovery_abs'] and
                       result.get('prediction',{}).get('relative_l2',math.inf)<=p['prediction_recovery_l2'])
    result['total_seconds_excluding_reference_imports_and_warmup']=time.perf_counter()-begin-result.get('warmup_seconds',0.)
    return result
