"""Read-only, portable integrity/pairing replay of the targeted evidence snapshot.

Uses Python's standard library. Does not train models or require PyTorch.
"""
import hashlib
import json
import math
from pathlib import Path

P1=Path(__file__).resolve().parents[1]
REV=P1/'revision_neurocomputing_20261016'
RUN=REV/'targeted_validation_recovery_20261025'
OLD=REV/'targeted_validation_20261023'
CLOSE=REV/'targeted_validation_closure_20261025'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(root,path): return (root/str(path).replace('\\','/')).resolve()


def leaves(value):
    if isinstance(value,list):
        for x in value:yield from leaves(x)
    else:yield float(value)


def main():
    for root in [OLD,RUN]:
        receipt=read(root/'freeze_receipt.json')
        for key,item in receipt['bindings'].items():
            assert sha(relative(root,item['path']))==item['sha256'],(root.name,key)
    reconciliation=read(CLOSE/'PROVENANCE_RECONCILIATION.json')
    for proof in reconciliation['proofs']:
        assert sha(P1/proof['original_receipt'])==proof['original_sha256']
        assert sha(REV/proof['experiment']/'freeze_receipt.json')==proof['relocated_sha256']
    prior=read(RUN/'prior_preservation.json')
    for rel,digest in prior['files'].items():
        if rel=='freeze_receipt.json':
            assert any(p['experiment']==OLD.name and p['original_sha256']==digest for p in reconciliation['proofs'])
        else:assert sha(relative(OLD,rel))==digest,rel
    audit=read(CLOSE/'AUDIT.json')
    for rel,digest in audit['raw_terminal_sha256'].items():assert sha(relative(REV,rel))==digest,rel
    retry=read(CLOSE/'io_retry/terminal.json')
    assert sha(CLOSE/'io_retry/terminal.json')==audit['retry_terminal_sha256']
    assert retry['status']=='completed' and retry['completed_updates']==100 and retry['recovery']
    rows=[read(p) for p in (RUN/'results').glob('*/terminal/*.json')]
    jobs=read(RUN/'schedule.json');planned={j['id']:j for j in jobs};actual={r['id']:r for r in rows}
    assert len(rows)==len(actual)==len(planned)==583
    for identity,job in planned.items():
        for key,val in job.items():assert actual[identity][key]==val,(identity,key)
    for item in read(RUN/'results/B/reuse.json')['records']:
        assert sha(relative(OLD,item['original_path']))==item['sha256']
        assert sha(RUN/'results/B/terminal'/(item['id']+'.json'))==item['sha256']
    pairs=0;p=read(RUN/'protocol.json')
    effective=dict(actual);effective[retry['id']]=retry
    for size in [4,8,16]:
        for regime in ['mild','hard']:
            for seed in [101,211,307]:
                left=effective[f'C-{size}-{regime}-{seed}-selective']
                right=effective[f'C-{size}-{regime}-{seed}-full_hp']
                assert left['status']==right['status']=='completed'
                assert left['completed_updates']==right['completed_updates']==100
                for step in p['paired_steps']:
                    for field in ['raw_before','raw_after','physical_before','physical_after','raw_gradient','loss']:
                        a=list(leaves(left['trajectory'][step-1][field]));b=list(leaves(right['trajectory'][step-1][field]))
                        assert len(a)==len(b)
                        assert all(math.isfinite(x) and abs(x-y)<=p['pair_atol']+p['pair_rtol']*abs(y) for x,y in zip(a,b)),(size,regime,seed,step,field)
                pairs+=1
    tests=0
    for f in (CLOSE/'verification_unrestricted').glob('*.json'):
        result=read(f);assert result['status']=='completed' and result['exit_code']==0,f.name
        import re
        tests+=int(re.search(r'Ran (\d+) tests?',result['stderr']).group(1))
    assert tests==127
    result=dict(integrity_and_pairing_replay_passed=True,terminal_records=583,
                reused_B_records=336,complete_paired_trajectories=pairs,recorded_regression_tests=tests,
                original_io_error_preserved=True,scientific_retraining_performed=False,
                fresh_environment_install_claimed=False,uncontended_runtime_claimed=False)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':main()
