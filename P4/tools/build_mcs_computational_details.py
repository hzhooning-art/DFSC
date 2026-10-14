"""Expose frozen grid-refinement evidence; no additional numerical solves."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/mcs_prediction_figures'
REL='results/p4_observation_weighted_prediction_20261011.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    result=json.loads((ROOT/REL).read_text(encoding='utf-8'))
    spec_path=ROOT/'experiments/specs/observation_weighted_prediction_20261011.json'
    spec=json.loads(spec_path.read_text(encoding='utf-8'))
    assert sha(spec_path)==result['spec_sha256']
    assert sha(ROOT/'experiments/observation_weighted_prediction.py')==result['source_sha256']
    assert sha((ROOT/REL).with_suffix('.npz'))==result['witness_archive_sha256']
    assert len(result['rows'])==80 and len(result['lp_logs'])==320
    summaries=[]
    for cells in spec['cell_counts']:
        rows=[r for r in result['rows'] if r['cells']==cells]
        assert len(rows)==16
        summaries.append({'cells':cells,'target_cases':16,
                          'max_endpoint_bracket':max(r['endpoint_bracket'] for r in rows),
                          'truths_contained':sum(r['contains_truth'] for r in rows)})
    assert all(s['truths_contained']==16 for s in summaries)
    assert all(summaries[i+1]['max_endpoint_bracket']<=s['max_endpoint_bracket'] for i,s in enumerate(summaries[:-1]))
    assert summaries[-1]['max_endpoint_bracket']<=spec['endpoint_bracket_tolerance']
    assert result['monotonicity_checks']==256
    lines=[r'\begin{table}[ht]',r'\centering\small',
           r'\caption{Saved grid refinement for the observation-weighted study. Each level reuses the same four spectra and four targets. The endpoint bracket is the maximum discrepancy between corresponding inner and outer endpoints, not prediction error against truth.}',
           r'\label{tab:mcs-grid}',r'\begin{tabular}{rrr}',r'\toprule',
           r'Cells & Maximum endpoint bracket & Targets contained\\\midrule']
    for s in summaries:lines.append(f"{s['cells']} & {s['max_endpoint_bracket']:.6f} & {s['truths_contained']}/16"+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}','']
    OUT.mkdir(parents=True,exist_ok=True)
    table=OUT/'grid_refinement_table.tex'
    table.write_text('\n'.join(lines),encoding='utf-8')
    report={'schema':'P4-MCS-computational-details-v1','all_checks_pass':True,
            'input_sha256':{REL:sha(ROOT/REL),'experiments/specs/observation_weighted_prediction_20261011.json':sha(spec_path)},
            'builder_sha256':sha(Path(__file__)),'generated_sha256':{table.name:sha(table)},
            'summaries':summaries,'saved_programs':320,'nested_endpoint_checks':256,
            'new_experiments':False,'new_solver_runs':False,
            'limits':'Retrospective summary of one saved refinement study; not certification of every capped/noisy experiment or floating-point containment.'}
    (OUT/'computational_details_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
