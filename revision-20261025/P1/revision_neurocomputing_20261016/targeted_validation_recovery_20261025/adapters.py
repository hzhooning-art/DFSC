"""Pinned, explicit adapters. No patches or detached-Parameter graph disguises."""
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from FDEint import FDEint
from pycaputo.controller import make_fixed_controller
from pycaputo.derivatives import CaputoDerivative
from pycaputo.events import StepCompleted
from pycaputo.fode import caputo
from pycaputo.stepping import evolve

from freeze_protocol import REV,WHEEL_SHA
from records import sha, exception_record
from references import finite, refined_derivative


def verify_source_bytes(path,expected):
    if Path(path).read_bytes()!=expected:
        raise ValueError('Frozen backend source differs from wheel: '+str(path))


@lru_cache(maxsize=1)
def load_frozen_backends():
    wheel=REV/'cpu_repair_delivery_20261021/output/dist/dfsc_cpu_repair_candidate-0.1.0.dev0-py3-none-any.whl'
    if sha(wheel)!=WHEEL_SHA:
        raise ValueError('Optional wheel hash mismatch')
    source=REV/'cpu_repair_delivery_20261021/backend/src/dfsc_cpu_repair'
    hashes={}
    with zipfile.ZipFile(wheel) as archive:
        for name in ('__init__.py','guarded_series.py','high_precision_cpu.py','selective_spectral.py'):
            verify_source_bytes(source/name,archive.read('dfsc_cpu_repair/'+name))
            hashes[name]=sha(source/name)
    namespace='_p1_targeted_frozen_repair'
    spec=importlib.util.spec_from_file_location(namespace,source/'__init__.py',submodule_search_locations=[str(source)])
    package=importlib.util.module_from_spec(spec); sys.modules[namespace]=package
    spec.loader.exec_module(package)
    spec_old=importlib.util.spec_from_file_location('_p1_targeted_original_ml',REV/'software_snapshot/dfsc/mittag_leffler.py')
    original=importlib.util.module_from_spec(spec_old); spec_old.loader.exec_module(original)
    return {'package':package,'original':original,'source_snapshot_execution':True,
            'module_hashes':hashes,'wheel_sha256':WHEEL_SHA}


def problem(name,alpha):
    if name=='scalar':
        operator=[[1.]]; u0=[[1.]]
    elif name=='rotated2':
        c,s=math.cos(.4),math.sin(.4)
        q=np.array([[c,-s],[s,c]])
        operator=(q@np.diag([.3,6.])@q.T).tolist(); u0=[[1.,-.25]]
    else:
        raise ValueError('Unknown common problem')
    return {'name':name,'operator':operator,'u0':u0,'times':[0,.125,.25,.5,.75,1.], 'alpha':alpha}


def connected_gradients(loss,parameters):
    gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
    if any(g is None for g in gradients):
        raise RuntimeError('Requested upstream parameter is not connected')
    if any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise ArithmeticError('Nonfinite backward gradient')
    return gradients


def build_action(method,operator):
    backend=load_frozen_backends()
    A=torch.as_tensor(operator,dtype=torch.float64)
    if method in ('selective','full_hp','guard'):
        mode={'selective':'selective_repair','full_hp':'all_high_precision','guard':'guard_only'}[method]
        return backend['package'].SelectiveSpectral(A,mode=mode)
    if method=='original':
        lam,q=torch.linalg.eigh(A)
        def action(u0,times,alpha,*,beta):
            safe=torch.where(times>0,times,torch.ones_like(times))
            powers=torch.where(times>0,safe.pow(alpha),torch.zeros_like(times))
            z=-powers[:,None]*lam[None,:].pow(beta/2)
            kernel=backend['original'].mittag_leffler_e(alpha,z,terms=180,method='hybrid')
            output=((u0@q)[:,None,:]*kernel[None,:,:])@q.T
            return {'values':output,'fast_entries':z.numel(),'repaired_entries':0,'total_entries':z.numel()}
        return action
    raise ValueError('No spectral action for '+method)


def action_values(result):
    if isinstance(result,dict):
        return result['values'],{k:result[k] for k in ('fast_entries','repaired_entries','total_entries')}
    return result.values,{k:getattr(result,k) for k in ('fast_entries','repaired_entries','total_entries')}


def forward(method,operator,u0,times,alpha,steps=None,need_grad=False):
    # Imports/source verification are setup exclusions; model/basis construction is timed.
    if method in ('selective','full_hp','guard','original'):
        load_frozen_backends()
    start=time.perf_counter()
    try:
        A=torch.as_tensor(operator,dtype=torch.float64)
        u=u0 if torch.is_tensor(u0) else torch.tensor(u0,dtype=torch.float64)
        u=u.reshape(-1,A.shape[0])
        t=torch.as_tensor(times,dtype=torch.float64)
        a=alpha if torch.is_tensor(alpha) else torch.tensor(alpha,dtype=torch.float64)
        if method in ('selective','full_hp','guard','original'):
            model=build_action(method,A)
            setup=time.perf_counter()-start
            values,counts=action_values(model(u,t,a,beta=torch.tensor(2.,dtype=torch.float64)))
            internal_steps=None
        elif method=='fdeint':
            h=torch.tensor(float(t.max())/steps,dtype=torch.float64)
            prepend_zero=bool(t[0]>0)
            solve_t=torch.cat([torch.zeros(1,dtype=torch.float64),t]) if prepend_zero else t
            setup=time.perf_counter()-start
            values,t_internal,_=FDEint(lambda now,state:-state@A.T,solve_t,u,a,h=h,
                                      dtype=torch.float64,DEBUG=False,return_internals=True)
            if prepend_zero:
                values=values[:,1:,:]
            internal_steps=t_internal.shape[1]-1; counts={}
        elif method=='pycaputo':
            if need_grad:
                raise ValueError('Selected NumPy pycaputo path has no native torch gradient; use explicit FD')
            array=A.detach().numpy(); starts=u.detach().numpy(); alpha_float=float(a.detach())
            dt=float(t.max())/steps
            setup=time.perf_counter()-start
            rows=[]; internal_steps=0
            for initial in starts:
                model=caputo.PECE(ds=(CaputoDerivative(alpha_float),)*A.shape[0],
                    control=make_fixed_controller(dt,tstart=0.,tfinal=float(t.max())),
                    source=lambda now,state:-array@state,y0=(initial.copy(),),corrector_iterations=1)
                events=[event for event in evolve(model,dtinit=dt) if isinstance(event,StepCompleted)]
                ts=np.array([float(ev.t) for ev in events]); ys=np.array([ev.y for ev in events])
                if ts[0]>0 or ts[-1]+1e-12<float(t.max()):
                    raise ArithmeticError('Stepper did not span requested observations')
                rows.append(np.stack([np.interp(t.numpy(),ts,ys[:,j]) for j in range(A.shape[0])],axis=-1))
                internal_steps+=len(events)-1
            values=torch.tensor(np.stack(rows),dtype=torch.float64); counts={}
        else:
            raise ValueError('Unknown method '+method)
        if not bool(torch.isfinite(values).all()):
            raise ArithmeticError('Nonfinite returned action')
        return dict(status='returned',values=values,elapsed_seconds=time.perf_counter()-start,
                    setup_seconds=setup,internal_steps=internal_steps,**counts)
    except Exception as exc:
        package=load_frozen_backends()['package'] if method in ('selective','full_hp','guard','original') else None
        refused=package is not None and isinstance(exc,(package.ActionRefusal,package.RepairRefusal))
        return dict(status='refused' if refused else 'error',elapsed_seconds=time.perf_counter()-start,**exception_record(exc))


def scalar_loss(method,p,alpha,steps,u0=None):
    a=torch.nn.Parameter(torch.tensor(float(alpha),dtype=torch.float64))
    row=forward(method,p['operator'],p['u0'] if u0 is None else u0,p['times'],a,steps,False)
    if row['status']!='returned':
        raise RuntimeError(str({k:v for k,v in row.items() if k!='values'}))
    return float(row['values'].detach().square().mean())


def loss_gradient(method,p,alpha_path,steps):
    result={'gradient_label':'explicit_discrete_FD' if method=='pycaputo' else 'native_torch_AD',
            'alpha_path':alpha_path,'status':'error'}
    alpha=p['alpha']
    if method!='pycaputo':
        raw_value=alpha if alpha_path=='physical' else math.log((alpha-.45)/(.95-alpha))
        raw=torch.nn.Parameter(torch.tensor(raw_value,dtype=torch.float64))
        physical=raw if alpha_path=='physical' else .45+.5*torch.sigmoid(raw)
        u0=torch.tensor(p['u0'],dtype=torch.float64,requires_grad=True)
        row=forward(method,p['operator'],u0,p['times'],physical,steps,True)
        if row['status']=='returned':
            try:
                ga,gu=connected_gradients(row['values'].square().mean(),(raw,u0))
                chain=1. if alpha_path=='physical' else float((.5*torch.sigmoid(raw)*(1-torch.sigmoid(raw))).detach())
                result.update(status='returned',alpha_gradient=float(ga)/chain,
                              raw_gradient=float(ga),chain=chain,u0_gradient=gu.tolist())
            except Exception as exc:
                result.update(exception_record(exc))
        else:
            result.update({k:v for k,v in row.items() if k!='values'})
    fd=refined_derivative(lambda a:scalar_loss(method,p,a,steps),alpha,[.01,.005,.0025])
    result['discrete_alpha_fd']=fd
    u=np.array(p['u0']); ufds=[]
    for i in range(u.shape[0]):
        row=[]
        for j in range(u.shape[1]):
            def loss_at(v):
                uu=u.copy(); uu[i,j]=v
                return scalar_loss(method,p,alpha,steps,uu)
            row.append(refined_derivative(loss_at,float(u[i,j]),[.01,.005,.0025]))
        ufds.append(row)
    result['discrete_u0_fd']=ufds
    if method=='pycaputo' and fd['status']=='resolved' and all(x['status']=='resolved' for row in ufds for x in row):
        result.update(status='returned',alpha_gradient=fd['value'],u0_gradient=[[x['value'] for x in row] for row in ufds],
                      native_torch_u0=False)
    elif method=='pycaputo':
        result['status']='reference_unresolved'
    result['discrete_alpha_comparison_status']=fd['status']
    result['discrete_u0_comparison_status']=[[x['status'] for x in row] for row in ufds]
    if result['status']=='returned':
        result['native_vs_discrete_alpha_error']=(abs(result['alpha_gradient']-fd['value'])
                                                 if fd['status']=='resolved' else None)
        result['native_vs_discrete_u0_errors']=[[
            abs(v-x['value']) if x['status']=='resolved' else None for v,x in zip(row,fs)]
            for row,fs in zip(result['u0_gradient'],ufds)]
    return result
