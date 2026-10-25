"""Retain transport decisions; refine only unresolved identical-profile targets."""
from copy import deepcopy
from prediction_time_search import decision


def polish_unresolved(info, gamma, result, absolute, relative, rho, levels, oracle, feasible):
    output = deepcopy(result)
    output['continuous_dual_extension'] = {'applicable':gamma == info.beta and feasible is not None,
        'attempted_targets':0,'resolved_sufficient':0,'resolved_insufficient':0,
        'additional_endpoint_LPs':0,'residual_cells':0}
    extension = output['continuous_dual_extension']
    if not extension['applicable']:
        return output
    for row in output['rows']:
        if row['state'] != 'unresolved':
            continue
        extension['attempted_targets'] += 1
        before = deepcopy(row)
        row['pre_correction'] = before
        row['dual_levels_used'] = []
        for level in levels:
            row['dual_levels_used'].append(level)
            try:
                bracket = oracle(row['wait'],level)
            except RuntimeError as error:
                programs = getattr(error,'programs',0)
                calls = getattr(error,'solver_calls',0)
                variables = programs*getattr(error,'cells',0)
                row['lp_calls'] += programs
                row['solver_calls'] += calls
                row['cell_variables'] += variables
                extension['additional_endpoint_LPs'] += programs
                row['dual_numerical_failure'] = str(error)
                break
            for k in ('lp_calls','solver_calls','cell_variables'):
                row[k] += (bracket['programs'] if k=='lp_calls' else bracket['solver_calls'] if k=='solver_calls'
                           else bracket['programs']*bracket['cells'])
            extension['additional_endpoint_LPs'] += bracket['programs']
            extension['residual_cells'] += sum(c['cells'] for c in bracket['certificates'].values())
            # Preserve older bounds if one component of the new bracket is looser.
            old_outer, old_inner = before.get('outer'), before.get('inner')
            outer = bracket['outer']
            inner = bracket['inner']
            if old_outer is not None:
                outer = [max(outer[0],old_outer[0]*(1+rho)),min(outer[1],old_outer[1]*(1-rho))]
            if old_inner is not None:
                inner = [min(inner[0],old_inner[0]*(1+rho)),max(inner[1],old_inner[1]*(1-rho))]
            value = decision({'outer':outer,'inner':inner},absolute,relative,rho)
            row.update(value,route='continuous_dual',dual_fast_rate=bracket['fast_rate'])
            if value['state'] != 'unresolved':
                extension['resolved_'+value['state']] += 1
                break
    for k in ('lp_calls','solver_calls','cell_variables'):
        output[k] = sum(r[k] for r in output['rows'])
    passing = [r['wait'] for r in output['rows'] if r['state']=='sufficient']
    first = min(passing) if passing else None
    output['first_sufficient_candidate'] = first
    output['earliest_candidate_proved'] = first is not None and all(
        r['state'] in ('insufficient','unbounded') for r in output['rows'] if r['wait'] < first)
    return output
