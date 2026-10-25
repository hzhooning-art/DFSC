"""Full-denominator evidence audit; missing/partial cells cannot become successes."""
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from external_comparison import qualifies
from freeze_protocol import HERE
from records import read,write_terminal,verify_bindings,sha


def complete_rows(jobs,rows):
    planned={j['id']:j for j in jobs}
    if len(planned)!=len(jobs):
        raise ValueError('Duplicate planned identity')
    found={}
    for row in rows:
        key=row['id']
        if key in found or key not in planned:
            raise ValueError('Duplicate/unknown terminal identity: '+key)
        for k,v in planned[key].items():
            if row.get(k)!=v:
                raise ValueError('Terminal input binding mismatch: '+key+'/'+k)
        found[key]=row
    return [found.get(j['id'],dict(j,status='not_run_budget')) for j in jobs]


def trajectory(row):
    return row.get('trajectory',[]) or (row.get('checkpoint') or {}).get('trajectory',[])


def compare_pairs(rows,p):
    groups=defaultdict(dict)
    for row in rows:
        if row.get('method') in ('selective','full_hp'):
            groups[(row['size'],row['regime'],row['seed'])][row['method']]=row
    pairs=[]
    for key,group in groups.items():
        pair={'size':key[0],'regime':key[1],'seed':key[2],'full_hundred_step_pair':False,'shared_steps':0,'checks':[]}
        if len(group)<2:
            pair['status']='missing_counterpart'; pairs.append(pair); continue
        selective,full=group['selective'],group['full_hp']
        st,ft=trajectory(selective),trajectory(full)
        shared=min(len(st),len(ft)); pair['shared_steps']=shared
        if not shared:
            pair['status']='no_shared_update'; pairs.append(pair); continue
        steps=sorted(set([n for n in p['paired_steps'] if n<=shared]+[shared]))
        for n in steps:
            fields=[]
            for field in ('raw_before','raw_after','physical_before','physical_after','raw_gradient','loss'):
                if field not in st[n-1] or field not in ft[n-1]:
                    continue
                sv,fv=np.asarray(st[n-1][field]),np.asarray(ft[n-1][field])
                fields.append({'field':field,'max_abs':float(np.max(np.abs(sv-fv))),
                               'pass':bool(np.all(np.abs(sv-fv)<=p['pair_atol']+p['pair_rtol']*np.abs(fv)))})
            pair['checks'].append({'step':n,'fields':fields,'pass':bool(fields) and all(f['pass'] for f in fields)})
        pair['status']='matched_observed_steps' if all(c['pass'] for c in pair['checks']) else 'trajectory_mismatch'
        pair['full_hundred_step_pair']=(shared==100 and all(r.get('status')=='completed' and r.get('is_hundred_step_run') for r in (selective,full)))
        if pair['full_hundred_step_pair']:
            s=selective.get('total_seconds_excluding_reference_imports_and_warmup')
            f=full.get('total_seconds_excluding_reference_imports_and_warmup')
            if s is not None and f is not None and s>0:
                pair['full_run_cost_ratio']=f/s
        pairs.append(pair)
    return pairs


def summarize(jobs,rows,p):
    rows=complete_rows(jobs,rows)
    tracks={}
    for track in ('A','B','C'):
        group=[r for r in rows if r['track']==track]
        tracks[track]={'planned':len(group),'status_counts':dict(Counter(r['status'] for r in group)),
                       'actually_attempted':sum(r['status']!='not_run_budget' for r in group)}
    a_rows=[r for r in rows if r['track']=='A']
    qualified=[]
    group=defaultdict(list)
    for row in a_rows:
        if row.get('kind')=='forward':
            group[(row['problem'],row['alpha'],row['method'])].append(row)
    for key,configs in sorted(group.items()):
        for band in p['accuracy_bands']:
            good=[c for c in configs if qualifies(c,band) and c.get('timing',{}).get('median_seconds',0)>0]
            if good:
                best=min(good,key=lambda c:c['timing']['median_seconds'])
                qualified.append({'problem':key[0],'alpha':key[1],'method':key[2],'band':band,
                    'job_id':best['id'],'steps':best.get('steps'),'median_seconds':best['timing']['median_seconds'],
                    'relative_l2':best['relative_l2'],'max_abs':best['max_abs'],
                    'interpretation':'minimum observed cost among measured qualifying settings, not universal ranking'})
    b_rows=[r for r in rows if r['track']=='B']; strata={}; repaired=[]
    for stratum in sorted({r['coordinate']['stratum'] for r in b_rows}):
        group=[r for r in b_rows if r['coordinate']['stratum']==stratum]
        methods={}
        for method in ('guard','full_hp','original'):
            outcomes=[]; ratios=[]
            for row in group:
                item=next((m for m in row.get('methods',[]) if m['method']==method),None)
                outcomes.append(item['outcome'] if item else row['status'])
                if item and item.get('outcome') in ('accurate_return','inaccurate_return'):
                    ratios.extend(item.get('error_target_ratios',[]))
                if method=='full_hp' and item and row['coordinate']['within_domain']:
                    repaired.append(item)
            methods[method]={'outcomes':dict(Counter(outcomes)), 'max_error_target_ratio':max(ratios) if ratios else None}
        strata[stratum]={'planned':len(group),'methods':methods,
                         'old_coordinate_overlaps':sum(r['coordinate'].get('overlaps_previous',False) for r in group)}
    c_rows=[r for r in rows if r['track']=='C']
    pairs=compare_pairs(c_rows,p)
    blocked=[r['id'] for r in rows if r.get('promotion_blocked')]
    for row in a_rows:
        if row.get('method') not in ('selective','full_hp'):
            continue
        if row['status']=='error' or (row.get('continuous_gradient_check') and not row['continuous_gradient_check']['all_pass']):
            blocked.append(row['id'])
        if row.get('kind')=='forward' and row.get('status')=='returned':
            vals=np.array(row['calls'][-1]['values']); ref=np.array(row['reference']['values'])
            if not bool((np.abs(vals-ref)<=1e-10+1e-8*np.abs(ref)).all()):
                blocked.append(row['id'])
    blocked.extend('pair:'+str((r['size'],r['regime'],r['seed'])) for r in pairs if r['status']=='trajectory_mismatch')
    successful_b=sum(r.get('outcome')=='accurate_return' for r in repaired)
    no_missing=bool(jobs) and all(r['status'] not in ('not_run_budget','timeout','interrupted','partial','reference_unresolved','error') for r in rows)
    return {'tracks':tracks,'a_accuracy_matched':qualified,'b_strata':strata,
        'a_inverse':{'planned':sum(r.get('kind')=='inverse' for r in a_rows),
                     'sixty_step_completed':sum(bool(r.get('is_sixty_step_run')) for r in a_rows),
                     'recovered':sum(bool(r.get('recovery')) for r in a_rows)},
        'b_useful_observed_repair':successful_b>0 and not any(r.get('promotion_blocked') for r in repaired),
        'b_repaired_accurate_return_count':successful_b,
        'b_laplace_selected':sum(r['coordinate'].get('laplace_selected',False) for r in b_rows),
        'b_laplace_statuses':dict(Counter(r.get('laplace',{}).get('status','not_executed') for r in b_rows if r['coordinate'].get('laplace_selected'))),
        'b_all_planned_coordinates_completed':len(b_rows)==336 and all(r['status']=='completed' for r in b_rows),
        'c_pairs':pairs,'c_hundred_step_completed':sum(bool(r.get('is_hundred_step_run')) for r in c_rows),
        'c_recovered':sum(bool(r.get('recovery')) for r in c_rows),
        'c_full_paired_matches':sum(r['full_hundred_step_pair'] and r['status']=='matched_observed_steps' for r in pairs),
        'promotion_blocked':bool(blocked),'promotion_blocking_cells':sorted(set(blocked)),
        'all_tracks_complete':no_missing and successful_b>0 and not blocked and
            len(tracks)==3 and all(tracks[t]['planned']>0 for t in tracks),
        'not_independent_material_replicates':True,'global_certificate':False,'paper_or_release_updated':False}


def audit(root=HERE):
    root=Path(root); verify_bindings(root)
    p=read(root/'protocol.json'); jobs=read(root/'schedule.json')
    files=sorted((root/'results').glob('*/terminal/*.json'))
    rows=[read(path) for path in files]
    full=complete_rows(jobs,rows)
    report=summarize(jobs,full,p)
    report['terminal_bindings']={str(path.relative_to(root)):sha(path) for path in files}
    report['protocol_sha256']=sha(root/'protocol.json')
    report['freeze_receipt_sha256']=sha(root/'freeze_receipt.json')
    report['all_terminal_files_present']=len(files)==len(jobs)
    preserved=read(root/'freeze_receipt.json').get('preserved_artifacts',{})
    report['preserved_previous_artifacts']=preserved
    report['previous_artifacts_unchanged']=all((root.parent/rel).is_file() and sha(root.parent/rel)==digest for rel,digest in preserved.items())
    report['track_ledgers']={t:read(root/'results'/t/'ledger.json') if (root/'results'/t/'ledger.json').exists() else None for t in ('A','B','C')}
    return report


def report_text(report):
    lines=['# Targeted validation: bounded evidence report','',
           'The CPU/SPD candidate and the preceding manuscript/delivery remain unchanged.',
           'A known spectral propagator is compared with history-aware steppers; the structural advantage is explicit.',
           'All counts describe selected numerical problems, not independent physical materials.','',
           '| Track | Planned | Attempted | Terminal statuses |','|---|---:|---:|---|']
    if report.get('primary_uncontended_runtime_claim_supported') is False:
        lines[2:2]=['**Runtime evidence disqualified: known background CPU contention. Raw timings are retained, not eligible performance comparisons.**','']
    for t,row in report['tracks'].items():
        lines.append(f'| {t} | {row["planned"]} | {row["actually_attempted"]} | {json.dumps(row["status_counts"])} |')
    lines+=['','## Numerical and practical gates','',
            f'- Promotion blocked: {report["promotion_blocked"]}; cells: {report["promotion_blocking_cells"]}.',
            f'- B repaired accurate returned points: {report["b_repaired_accurate_return_count"]}; useful observed repair gate: {report["b_useful_observed_repair"]}.',
            f'- B all 336 coordinates completed: {report["b_all_planned_coordinates_completed"]}; preselected Laplace statuses: {report["b_laplace_statuses"]}.',
            f'- A inverse: {report["a_inverse"]}.',
            f'- C 100-step completions: {report["c_hundred_step_completed"]}; full paired trajectory matches: {report["c_full_paired_matches"]}; recoveries: {report["c_recovered"]}.',
            '- Timed forward comparisons use one warmup and three fresh calls; medians are descriptive, not confidence intervals.',
            '- Accuracy-qualified costs are minima only among measured settings satisfying both error criteria. Missing/failed settings cannot qualify.',
            '- Partial paired trajectories are not full 100-step evidence. Missing predictions are not successful recovery.','',
            '## Complete tables','', 'See `results/audit.json` for all strata, qualified configurations, paired trajectories and immutable terminal hashes.',
            'See `results/<track>/attempts/*/process.json` for stdout/stderr/timeout and `checkpoint.json` for completed updates.',
            'Censoring can prevent the intended full comparison; it is not a basis to rank only survivors.','',
            '## Unchanged limits','',
            'No global numerical certificate, GPU repair, general nonnormal/forced solver extension, new physical replicates or universal prediction gain follows.',
            'Full reviewer reports, approved author materials, external artifact access and portal approval remain separate requirements.',
            'No manuscript, reviewer response, public package, DOI or GitHub release was changed by this experiment.','']
    return '\n'.join(lines)


if __name__=='__main__':
    report=audit()
    write_terminal(HERE/'results/audit.json',report)
    with (HERE/'RESULTS_REPORT.md').open('x',encoding='utf-8') as handle:
        handle.write(report_text(report))
    print(json.dumps({k:v for k,v in report.items() if k in ('tracks','promotion_blocked','a_inverse','b_repaired_accurate_return_count','c_hundred_step_completed','c_full_paired_matches')}))
