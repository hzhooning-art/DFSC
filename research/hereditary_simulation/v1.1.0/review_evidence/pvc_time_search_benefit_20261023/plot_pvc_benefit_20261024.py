"""Render only frozen matched-comparison records; do not solve or retune."""
from pathlib import Path
from collections import defaultdict
import json, hashlib, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--result',type=Path)
parser.add_argument('--output-dir',type=Path)
parser.add_argument('--sync-submission',action='store_true')
args=parser.parse_args()
default=ROOT/'results/pvc_time_search_benefit_20261023/result.json'
path=args.result or (default if default.exists() else Path(__file__).with_name('result.json'))
assert hashlib.sha256(path.read_bytes()).hexdigest()=='53f5579a09d43e76f28dc50670e7d8fc01f924edf8a4ef63f3a6b3cc3c7c6462'
r=json.loads(path.read_text()); rows=defaultdict(lambda:[0,0])
for row in r['by_case']:
    temp=int(row['case'].split('_')[0][1:]);rows[temp][0]+=row['pointwise_endpoint_programs'];rows[temp][1]+=row['reuse_endpoint_programs']
assert sum(v[0] for v in rows.values())==650 and sum(v[1] for v in rows.values())==234
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10.2,3.65))
gray,teal='#87989e','#16718a';x=np.arange(len(rows));width=.34
for j,(color,label) in enumerate(((gray,'Pointwise continuous'),(teal,'Temporal reuse'))):
    bars=axes[0].bar(x+(j-.5)*width,[v[j] for v in rows.values()],width,color=color,label=label)
    axes[0].bar_label(bars,padding=3,fontsize=9)
axes[0].set(xticks=x,xticklabels=list(rows),xlabel='Temperature (K)',ylabel='Endpoint programs',ylim=(0,165),title='(a) Computation by temperature')
budgets=[b['endpoint_program_budget_per_case'] for b in r['budget_results']]
for name,color,marker in [('pointwise_classical',gray,'o'),('loading_guided_reuse',teal,'s')]:
    values=[b[name]['sufficient']+b[name]['insufficient'] for b in r['budget_results']]
    axes[1].plot(budgets,values,color=color,marker=marker,lw=2,markersize=5)
axes[1].set(xscale='log',xticks=budgets,xticklabels=budgets,ylim=(0,335),xlabel='Endpoint-program budget per workload',ylabel='Decided requests / 315',title='(b) Decisions within budget')
axes[1].axhline(310,color='#555555',ls=':',lw=1,zorder=0)
axes[1].text(4.1,317,'Final: 310 decided, 5 unresolved',fontsize=8,color='#444444')
for ax in axes: ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.015),ncol=2,frameon=False)
fig.subplots_adjust(top=.78,bottom=.19,wspace=.31,left=.07,right=.99)
out=args.output_dir or (ROOT/'output/figures' if default.exists() else Path(__file__).parent/'rendered_figures')
out.mkdir(parents=True,exist_ok=True)
for suffix in ('pdf','png'):fig.savefig(out/f'pvc_time_search_benefit_20261024.{suffix}',dpi=240,bbox_inches='tight')
plt.close(fig)
if args.sync_submission:
    for target in (ROOT/'build/jcam_source_20261024_pvc/Figures/Figure_10.pdf',ROOT/'submission_jcam_current/Figures/Figure_10.pdf'):
        target.write_bytes((out/'pvc_time_search_benefit_20261024.pdf').read_bytes())
print(json.dumps({'endpoint_programs_by_temperature':dict(rows),'budgets':budgets,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
