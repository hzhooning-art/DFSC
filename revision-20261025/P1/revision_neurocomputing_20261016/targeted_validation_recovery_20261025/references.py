"""Independent validation mathematics. Never imports production/admission code."""
import math
import mpmath
import numpy as np
import scipy.linalg
from pymittagleffler import mittag_leffler

from records import exception_record


def finite(value):
    array=np.asarray(value,dtype=float)
    if not np.isfinite(array).all():
        raise ArithmeticError('Nonfinite independent reference')
    return array


def value(alpha,z):
    answer=complex(mittag_leffler(float(z),float(alpha),1.))
    if not math.isfinite(answer.real) or not math.isfinite(answer.imag) or abs(answer.imag)>1e-9:
        raise ArithmeticError('Invalid independent Mittag-Leffler value: '+str(answer))
    return answer.real


def defining_sum(alpha,radius,digits,max_terms=6000):
    ctx=mpmath.mp.clone(); ctx.dps=digits
    # mpf(float) matches the EXACT binary float input; not mpf(str(float)).
    a,z=ctx.mpf(float(alpha)),-ctx.mpf(float(radius))
    if z==0:
        return ['1','0',ctx.nstr(1/ctx.gamma(a+1),digits-10)],0
    v,da,dz=ctx.mpf(1),ctx.mpf(0),ctx.mpf(0)
    consecutive=0; threshold=ctx.power(10,-(digits-20))
    for k in range(1,max_terms+1):
        coefficient=1/ctx.gamma(a*k+1)
        term=z**k*coefficient
        ta=-k*ctx.digamma(a*k+1)*term
        tz=k*z**(k-1)*coefficient
        v+=term; da+=ta; dz+=tz
        consecutive=consecutive+1 if max(abs(term),abs(ta),abs(tz))<threshold else 0
        if consecutive>=8:
            return [ctx.nstr(x,digits-10) for x in (v,da,dz)],k
    raise ArithmeticError('Independent defining reference exceeded term budget')


def refined_derivative(evaluator,x,steps):
    rows=[]
    try:
        for h in steps:
            raw=[finite(evaluator(x+c*h)) for c in (-2,-1,1,2)]
            d=(raw[0]-8*raw[1]+8*raw[2]-raw[3])/(12*h)
            rows.append({'step':h,'evaluations':[v.tolist() for v in raw], 'derivative':finite(d).tolist()})
        extrap=[(16*np.asarray(rows[i+1]['derivative'])-np.asarray(rows[i]['derivative']))/15
                for i in range(len(rows)-1)]
        if len(extrap)<2:
            raise ValueError('At least three FD steps required')
        gap=np.abs(extrap[-1]-extrap[-2])
        target=2e-8+2e-6*np.abs(extrap[-1])
        return {'status':'resolved' if bool((gap<=target).all()) else 'reference_unresolved',
                'rows':rows, 'richardson':[finite(v).tolist() for v in extrap],
                'value':finite(extrap[-1]).tolist(),'gap':gap.tolist(),'target':target.tolist()}
    except Exception as exc:
        return dict(status='reference_unresolved',rows=rows,**exception_record(exc))


def scalar_reference(alpha,radius,settings):
    result={'alpha':alpha,'radius':radius,'input_conversion':'mpf(float), exact float64 input',
            'digits':settings['reference_digits'],'status':'reference_unresolved','precision_runs':[]}
    try:
        for digits in settings['reference_digits']:
            values,terms=defining_sum(alpha,radius,digits,settings['reference_max_terms'])
            result['precision_runs'].append({'digits':digits,'values':values,'terms':terms})
        ctx=mpmath.mp.clone(); ctx.dps=max(settings['reference_digits'])
        lower,higher=[row['values'] for row in result['precision_runs']]
        agree=all(abs(ctx.mpf(lo)-ctx.mpf(hi))<=ctx.mpf(settings['reference_pair_rtol'])*(1+abs(ctx.mpf(hi)))
                  for lo,hi in zip(lower,higher))
        result.update(precision_agreement=agree,values=[float(ctx.mpf(v)) for v in higher])
        finite(result['values'])
        if not agree:
            return result
        independent=value(alpha,-radius)
        da=refined_derivative(lambda a:value(a,-radius),alpha,settings['fd_steps'])
        dz=refined_derivative(lambda z:value(alpha,z),-radius,settings['fd_steps'])
        result['independent']={'value':independent,'alpha':da,'z':dz}
        if da['status']!='resolved' or dz['status']!='resolved':
            return result
        external=[independent,da['value'],dz['value']]
        targets=[settings['independent_value_atol']+settings['independent_value_rtol']*abs(result['values'][0])]+[
            settings['independent_gradient_atol']+settings['independent_gradient_rtol']*abs(v) for v in result['values'][1:]]
        errors=[abs(v-ref) for v,ref in zip(external,result['values'])]
        result['independent'].update(errors=errors,targets=targets)
        if all(e<=t for e,t in zip(errors,targets)):
            result['status']='resolved'
    except Exception as exc:
        result.update(exception_record(exc))
    return result


def laplace_reference(alpha,radius,settings,*,inverter=None):
    runs=[]
    for digits in settings['laplace_digits']:
        ctx=mpmath.mp.clone(); ctx.dps=digits
        a,x=ctx.mpf(float(alpha)),ctx.mpf(float(radius))
        transform=[lambda s:s**(a-1)/(s**a+x),
                   lambda s:x*s**(a-1)*ctx.log(s)/(s**a+x)**2,
                   lambda s:s**(a-1)/(s**a+x)**2]
        row={'digits':digits,'method':'dehoog','outputs':[]}
        try:
            invert=ctx.invertlaplace if inverter is None else inverter
            for fn in transform:
                out=invert(fn,ctx.mpf(1),method='dehoog')
                if not ctx.isfinite(out):
                    raise ArithmeticError('Nonfinite inverse Laplace')
                row['outputs'].append(ctx.nstr(out,digits-8))
            row['status']='returned'
        except Exception as exc:
            row.update(status='error',**exception_record(exc))
        runs.append(row)
    if radius==0:
        return {'status':'analytical_origin_not_laplace_success','runs':runs,
                'analytical_values':[1.,0.,1/math.gamma(alpha+1)]}
    good=all(r['status']=='returned' for r in runs)
    ctx=mpmath.mp.clone(); ctx.dps=max(settings['laplace_digits'])
    if good:
        good=all(abs(ctx.mpf(lo)-ctx.mpf(hi))<=ctx.mpf('1e-35')*(1+abs(ctx.mpf(hi)))
                 for lo,hi in zip(runs[0]['outputs'],runs[1]['outputs']))
    return {'status':'resolved' if good else 'reference_unresolved','runs':runs,
            'values':[float(v) for v in runs[-1]['outputs']] if good else None}


def independent_action(operator,u0,times,alpha,beta=2.):
    lam,q=scipy.linalg.eigh(finite(operator))
    times=finite(times); u0=np.atleast_2d(finite(u0))
    radii=times[:,None]**float(alpha)*lam[None,:]**(float(beta)/2)
    kernels=np.array([[value(alpha,-x) for x in row] for row in radii])
    return finite(((u0@q)[:,None,:]*kernels[None,:,:])@q.T)


def action_reference(operator,u0,times,alpha,settings):
    result={'status':'reference_unresolved','modal_checks':[]}
    try:
        lam,q=scipy.linalg.eigh(finite(operator))
        u0=np.atleast_2d(finite(u0)); times=finite(times); beta=settings.get('beta',2.)
        radii=times[:,None]**float(alpha)*lam[None,:]**(beta/2)
        cache={}
        for x in radii.reshape(-1):
            if float(x) not in cache:
                cache[float(x)]=scalar_reference(float(alpha),float(x),settings)
                result['modal_checks'].append(cache[float(x)])
        if any(r['status']!='resolved' for r in result['modal_checks']):
            return result
        kernel=np.array([[value(alpha,-x) for x in row] for row in radii])
        y=((u0@q)[:,None,:]*kernel[None,:,:])@q.T
        loss_derivative=refined_derivative(lambda a:float(np.mean(independent_action(operator,u0,times,a,beta)**2)),
                                          alpha,settings['fd_steps'])
        gu=np.einsum('btm,tm->bm',(2*y/y.size)@q,kernel)@q.T
        result.update(status=loss_derivative['status'],values=finite(y).tolist(),
                      u0_gradient=finite(gu).tolist(),alpha_loss_derivative=loss_derivative,
                      loss=float(np.mean(y*y)),beta=beta)
    except Exception as exc:
        result.update(exception_record(exc))
    return result
