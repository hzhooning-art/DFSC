"""Publish all matched fit outcomes, including accurate baseline predictions."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from direct_fitting_baselines import ROOT,OUT,SPEC,sha

ARM_NAMES={'grid_ls':'Grid LS','grid_ridge':'Grid ridge','prony_cv':'Free-rate Prony',
           'mid_symmetric':'Symmetric midpoint','mid_classical':'Classical midpoint'}


def main():
    result=json.loads((OUT/'full.json').read_text());audit=json.loads((OUT/'full_audit.json').read_text())
    assert audit['all_checks_pass'] and audit['result_sha256']==sha(OUT/'full.json')
    protocol=json.loads(SPEC.read_text());rows=result['records']
    diagnostics={}
    for arm in ARM_NAMES:
        good=[r for r in rows if r['evaluation'][arm]['success']]
        diagnostics[arm]={'window_records':len(rows),'successes':len(good),
            'joint_exceedances':sum(any(r['evaluation'][arm]['exceeds_tolerance'].values()) for r in good),
            'paired_case_any_window_exceedances':len({r['case'] for r in good if any(r['evaluation'][arm]['exceeds_tolerance'].values())}),
            'paired_case_any_window_fit_failures':len({r['case'] for r in rows if not r['evaluation'][arm]['success']}),
            'band_compatible_but_target_exceeds':sum(r['evaluation'][arm].get('fit_band_compatible',False)
                     and any(r['evaluation'][arm]['exceeds_tolerance'].values()) for r in good),
            'band_compatible_outside_classical':sum(r['evaluation'][arm].get('fit_band_compatible',False)
                     and not all(r['evaluation'][arm].get('inside_classical_interval',{}).values()) for r in good)
                     if arm in ('grid_ls','grid_ridge','prony_cv') else None}
    report={'result_sha256':sha(OUT/'full.json'),'audit_sha256':sha(OUT/'full_audit.json'),
        'protocol_sha256':sha(SPEC),'diagnostics':diagnostics,'selected_orders':{},'selected_penalties':{}}
    for name,key,field in [('prony_cv','selected_orders','order'),('grid_ridge','selected_penalties','penalty')]:
        values=[r['models'][name].get(field) for r in rows if field in r['models'][name]]
        report[key]={str(v):values.count(v) for v in sorted(set(values))}
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    table=[r'\begin{table}[ht]',r'\centering\small',
        r'\caption{Direct point prediction on all 256 frozen cases per window. Mean absolute errors (MAE) condition on successful final fits. Joint exceedance means either absolute error exceeds $0.10$ for $G(20)$ or $0.05$ for held-out stress. These correlated window records are not independent trials. Midpoints are point estimates, not fitted models.}',
        r'\label{tab:direct-fits}',r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Window & Estimate & Success & MAE $G$ & MAE stress & Joint exceed\\\midrule']
    for h in protocol['windows']:
        summary=result['summaries'][f'all|{h}|all']
        for arm,label in ARM_NAMES.items():
            s=summary[arm];mae=s['mean_absolute_error']
            table.append(f"{h:g} & {label} & {s['success_count']}/256 & {mae['G20']:.5f} & {mae['heldout_sigma20']:.5f} & {s['joint_exceedance_count']}/{s['success_count']}"+r'\\')
        if h==8.:table.append(r'\midrule')
    table.extend([r'\bottomrule\end{tabular}',r'\end{table}',''])
    target=ROOT/'paper/mcs_prediction_figures/direct_fitting_table.tex'
    target.write_text('\n'.join(table))
    text=['# Direct fitting baseline results (14 October 2026)','',
        'New fits use saved observations; no new random trials, material measurements or independent blind cohort.',
        'The protocol was fixed before fitting, but existing target results were already known. Pilot cases are included in the full set and do not add independent trials.',
        'The same 80 responses, known unit mass, positive spectrum, rate cap 4 and exact loadings are used. Fit API accepts only observations, loading geometry and fixed protocol.',
        'Prony order and ridge penalty are selected on the trailing quarter of each loading record, then refitted. Fixed numerical grids are not continuum truth assumptions.',
        'Absolute point-error limits are 0.10/0.05, half the original diameter limits; this does not alter the old prospective width decisions.',
        'Point-baseline intervals are NOT supplied or asserted calibrated. Mean point error does not establish worst-case uncertainty superiority.','',
        '## All-case results','',
        '| Window | Method | Successful fits | Mean absolute G error | Mean absolute stress error | Joint tolerance exceedances among successes |',
        '|---|---|---:|---:|---:|---:|']
    for h in protocol['windows']:
        for arm,label in ARM_NAMES.items():
            s=result['summaries'][f'all|{h}|all'][arm];mae=s['mean_absolute_error']
            text.append(f"| {h:g} | {label} | {s['success_count']}/256 | {mae['G20']:.6f} | {mae['heldout_sigma20']:.6f} | {s['joint_exceedance_count']}/{s['success_count']} |")
    text += ['','## Family-stratified results','',
        '| Family | Window | Method | Success | Mean G error | Mean stress error | Joint exceedances |',
        '|---|---|---|---:|---:|---:|---:|']
    for family in sorted({r['family'] for r in rows}):
        for h in protocol['windows']:
            for arm,label in ARM_NAMES.items():
                s=result['summaries'][f'{family}|{h}|all'][arm];mae=s['mean_absolute_error']
                text.append(f"| {family} | {h:g} | {label} | {s['success_count']}/64 | {mae['G20']:.6f} | {mae['heldout_sigma20']:.6f} | {s['joint_exceedance_count']}/{s['success_count']} |")
    text += ['','## Calibration and failures','',
        'All 12 calibration-underestimation cases remain. The full JSON reports all/valid/invalid strata separately; the true validity flag never enters fitting.',
        f"Primary fitting wall time: {result['duration_seconds']:.2f} s with {result.get('workers',1)} workers. CPU occupancy is not converted into a fair LP-vs-fitting timing claim.",
        'Independent-primitive replay: maximum prediction discrepancy '+str(audit['maximum_independent_prediction_gap'])+'.',
        'Final refit failures: '+json.dumps(audit['final_refit_failures'])+'.',
        'Order/penalty candidate failures: '+json.dumps(audit['candidate_order_or_penalty_failures'])+'.',
        'Failed nonlinear starts: '+str(audit['failed_nonlinear_starts'])+'.',
        'These failures are retained; they are not evidence of inherent Prony failure or global optimality of successful fits.',
        'Selected order distribution: '+json.dumps(report['selected_orders'])+'.',
        'Selected penalty distribution: '+json.dumps(report['selected_penalties'])+'.','',
        '## Decision implications','',
        'A small response residual alone is not a target certificate. Nevertheless, this dataset is often already information-sufficient, so accurate fitting results must be retained. No general fitting-error, confidence-coverage, runtime or physical-validity advantage is certified.',
        'The direct point-prediction gap is addressed within this synthetic information class. Calibrated fitting intervals, robust global nonlinear optimization and real-material validation remain separate gaps.','',
        'Reproduce in a fresh extraction: run the five tests, then direct_fitting_baselines.py --phase pilot, audit_direct_fitting_baselines.py --phase pilot, run_direct_fitting_full.py, audit_direct_fitting_baselines.py --phase full and report_direct_fitting_baselines.py. Saved-result audits can be run without rerunning fits. Outputs are guarded against overwriting.']
    doc=ROOT/'docs/P4_DIRECT_FITTING_RESULTS_20261014.md';doc.write_text('\n'.join(text)+'\n')
    report['generated_sha256']={str(f.relative_to(ROOT)):sha(f) for f in (target,doc)}
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
