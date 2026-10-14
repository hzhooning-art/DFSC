"""Retrospective decision/resource analysis of frozen P4 simulation records."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / 'paper/mcs_prediction_figures'
SPEC = ROOT / 'experiments/specs/mcs_saved_results_analysis_20261013.json'
RESULT = ROOT / 'results/p4_mcs_saved_results_analysis_20261013.json'
NAMES = ['ultraslow_two', 'mixed_four', 'uniform_slow', 'logwide_continuous']
LABELS = ['Two-mode', 'Four-mode', 'Slow uniform', 'Log-wide']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(row, limits):
    if all(row['diameter']['outer'][t] <= v for t, v in limits.items()):
        return 'sufficient'
    if any(row['diameter']['inner'][t] > v for t, v in limits.items()):
        return 'insufficient'
    return 'unresolved'


def main():
    spec = json.loads(SPEC.read_text(encoding='utf-8'))
    rp = ROOT / 'results' / spec['parent_result']
    cp = ROOT / 'results' / spec['comparator_result']
    parent = json.loads(rp.read_text(encoding='utf-8'))
    comp = json.loads(cp.read_text(encoding='utf-8'))
    assert parent['case_count'] == 256 and parent['valid_calibration_count'] == 244
    assert sha(rp.with_suffix('.npz')) == parent['witness_archive_sha256']
    assert sha(cp.with_suffix('.npz')) == comp['witness_archive_sha256']
    for stem, data in [('ultraslow_calibrated_validation', parent), ('matched_convex_baseline', comp)]:
        audit = json.loads((ROOT / f'results/p4_{stem}_audit_20261011.json').read_text())
        assert audit['status'] == 'PASS'
        assert audit['result_sha256'] == sha(ROOT / f'results/p4_{stem}_20261011.json')
    limits = spec['utility_limits']
    parent_spec = json.loads((ROOT/'experiments/specs/ultraslow_calibrated_validation_20261011.json').read_text())
    assert limits == parent_spec['utility_limits']
    rows = {(r['case'], r['horizon']): r for r in parent['rows']}
    assert len(rows) == 512 and len(parent['cases']) == 256
    records, summaries = [], {}
    for name in NAMES:
        cases = [c for c in parent['cases'] if c['truth'] == name]
        counts = {str(h): {scope: {d: 0 for d in ['sufficient', 'insufficient', 'unresolved']}
                          for scope in ['all', 'valid', 'invalid']} for h in spec['horizons']}
        transition = {a+'->'+b: 0 for a in ['sufficient','insufficient','unresolved']
                      for b in ['sufficient','insufficient','unresolved']}
        for case in cases:
            states = []
            for h in spec['horizons']:
                row = rows[(case['id'], h)]
                assert row['truth'] == name and row['calibration_valid'] == case['calibration_valid']
                state = classify(row, limits)
                counts[str(h)]['all'][state] += 1
                counts[str(h)]['valid' if case['calibration_valid'] else 'invalid'][state] += 1
                states.append(state)
            if case['calibration_valid']:
                transition['->'.join(states)] += 1
            records.append({'case': case['id'], 'truth': name,
                            'calibration_valid': case['calibration_valid'],
                            'short_decision': states[0], 'long_decision': states[1]})
        s = parent['summaries'][name]
        summaries[name] = {'trials': len(cases), 'valid': s['valid_calibration_count'],
                           'counts': counts, 'valid_transitions': transition,
                           'conservative_combined_width_gain_percent':
                           100*(1-s['long_outer_to_short_inner_mean_ratio_valid_calibration'])}
    totals = {str(h): {scope: {d: sum(s['counts'][str(h)][scope][d] for s in summaries.values())
                    for d in ['sufficient','insufficient','unresolved']}
                    for scope in ['all','valid','invalid']} for h in spec['horizons']}
    assert totals['8.0']['valid'] == {'sufficient': 238, 'insufficient': 6, 'unresolved': 0}
    assert totals['16.0']['valid'] == {'sufficient': 244, 'insufficient': 0, 'unresolved': 0}
    assert totals['8.0']['all']['sufficient'] == 250
    assert sum(s['valid_transitions']['insufficient->sufficient'] for s in summaries.values()) == 6
    assert sum(s['valid_transitions']['sufficient->insufficient'] for s in summaries.values()) == 0
    FIGS.mkdir(exist_ok=True)
    table = [r'\begin{table}[ht]\centering\small',
             r'\caption{Retrospective resource/decision analysis using the original diameter limits. Counts condition on valid calibration for offline evaluation. No target or threshold is changed.}',
             r'\label{tab:mcs-resource}',
             r'\begin{tabular}{lrrrr}\toprule',
             r'Spectrum & Valid & Window 8 & Window 16 & Newly sufficient\\\midrule']
    for name, label in zip(NAMES,LABELS):
        s = summaries[name]
        table.append(f"{label} & {s['valid']} & {s['counts']['8.0']['valid']['sufficient']} & {s['counts']['16.0']['valid']['sufficient']} & {s['valid_transitions']['insufficient->sufficient']}"+r'\\')
    table += [r'\midrule Total & 244 & 238 & 244 & 6\\',
              r'\bottomrule\end{tabular}', r'\end{table}']
    (FIGS/'resource_decisions_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10,
                        'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
    fig,axes = plt.subplots(1,2,figsize=(9.4,3.6))
    x=np.arange(4)
    for i,h in enumerate(['8.0','16.0']):
        fraction=[summaries[n]['counts'][h]['valid']['sufficient']/summaries[n]['valid'] for n in NAMES]
        axes[0].bar(x+(i-.5)*.33,fraction,.31,label='Window '+h[:-2],color=['#89999e','#146a83'][i])
    axes[0].set_xticks(x,LABELS,rotation=15);axes[0].set_ylim(0,1.15)
    axes[0].set_ylabel('Fraction meeting both diameter limits')
    axes[0].set_title('Valid-calibration subset',pad=9)
    for i,name in enumerate(NAMES):
        s=summaries[name]
        axes[0].text(i,1.055,f"{s['counts']['8.0']['valid']['sufficient']}/{s['valid']} to {s['valid']}/{s['valid']}",ha='center',fontsize=7)
    duration=[spec['response_duration_sums'][h] for h in ['8.0','16.0']]
    axes[1].bar(['Window 8','Window 16'],duration,color=['#89999e','#146a83'],width=.48)
    axes[1].set_ylim(0,48);axes[1].set_title('Response acquisition resource')
    axes[1].set_ylabel('Sum of response-run durations')
    for i,n in enumerate(duration):axes[1].text(i,n+1,str(n),ha='center')
    axes[1].set_xlabel('80 responses and 64 blanks in each scenario',fontsize=8)
    fig.subplots_adjust(left=.085,right=.985,bottom=.30,top=.86,wspace=.32)
    handles,labels=axes[0].get_legend_handles_labels()
    legend=fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.015),ncol=2,frameon=False,fontsize=9)
    fig.canvas.draw()
    assert all(not legend.get_window_extent().overlaps(ax.get_window_extent()) for ax in axes)
    fig.savefig(FIGS/'resource_decisions.pdf',metadata={'CreationDate':None})
    fig.savefig(FIGS/'resource_decisions.png',dpi=300)
    plt.close(fig)
    result={'schema':'P4-MCS-saved-results-analysis-v1','date':'2026-10-13',
            'retrospective':True,'new_trials':False,'new_solvers':False,
            'spec_sha256':sha(SPEC),'source_sha256':sha(Path(__file__)),
            'parent_result_sha256':sha(rp),'comparator_result_sha256':sha(cp),
            'case_count':256,'valid_count':244,'summaries':summaries,'totals':totals,
            'duration_proxy_increase_percent':100*(40/24-1),
            'conditional_decision_gain_percentage_points':100*(244-238)/244,
            'paired_records':records,
            'generated_sha256':{name:sha(FIGS/name) for name in ['resource_decisions.pdf','resource_decisions_table.tex']},
            'limits':'Secondary analysis of existing simulations; not an adaptive-policy experiment, independent sample expansion, full cost assessment or new solver comparison.'}
    RESULT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'case_count':256,'valid':244,'short_sufficient':238,'long_sufficient':244,
                      'decision_gain_pp':result['conditional_decision_gain_percentage_points'],
                      'duration_proxy_increase_percent':result['duration_proxy_increase_percent']}))


if __name__=='__main__':
    main()
