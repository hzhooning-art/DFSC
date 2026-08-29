from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'results'/'stage66_extension_combination.json').read_text())
plt.rcParams.update({'font.family':'serif','font.size':11,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(10.5,7.2),constrained_layout=True)
colors={'base':'#446C9E','extension':'#D98C10','refusal':'#7A5195','ok':'#2A9D8F'}

def paired(ax,records,a,b,labels,title):
    x=np.arange(len(records)); va=np.array([r[a] for r in records]); vb=np.array([r[b] for r in records])
    for i in x: ax.plot([0,1],[va[i],vb[i]],color='#B8C1CC',lw=.8,zorder=1)
    ax.scatter(np.zeros_like(x),va,color=colors['base'],s=24,label=labels[0],zorder=2)
    ax.scatter(np.ones_like(x),vb,color=colors['extension'],s=24,label=labels[1],zorder=2)
    ax.set_xticks([0,1],labels); ax.set_yscale('log'); ax.set_ylabel('Held-horizon NRMSE'); ax.set_title(title,pad=9); ax.grid(axis='y',alpha=.25)

paired(axs[0,0],data['oscillatory']['records'],'real_nrmse','oscillatory_nrmse',('Real poles','Conjugate pair'),'Oscillatory-memory extension')
paired(axs[0,1],data['partial_sharing']['records'],'pooled_nrmse','partial_nrmse',('Fully shared','Partially shared'),'Group heterogeneity extension')
cont=np.array([r['dense_relative_gain'] for r in data['continuous_spectrum']['records']]); axs[1,0].scatter(np.arange(1,len(cont)+1),cont,color=colors['refusal'],s=30); axs[1,0].axhline(.20,color='black',ls='--',lw=1.1,label='Refusal threshold'); axs[1,0].set(xlabel='Seed',ylabel='Dense-grid relative gain',title='Continuous-spectrum stress test'); axs[1,0].set_ylim(0,max(.75,cont.max()+.05)); axs[1,0].legend(frameon=False); axs[1,0].grid(axis='y',alpha=.25)
conf=data['group_conformal']['records']; null=np.array([r['null_false_alarm_rate'] for r in conf]); shifted=np.array([r['shift_detection_rate'] for r in conf]); xpos=np.r_[np.zeros(len(null)),np.ones(len(shifted))]; vals=np.r_[null,shifted]; jitter=np.tile(np.linspace(-.08,.08,len(null)),2); axs[1,1].scatter(xpos+jitter,vals,c=[colors['ok']]*len(null)+[colors['extension']]*len(shifted),s=30); axs[1,1].plot([-.25,.25],[null.mean()]*2,color='black',lw=2); axs[1,1].plot([.75,1.25],[shifted.mean()]*2,color='black',lw=2); axs[1,1].axhline(.05,color='#666666',ls='--',lw=1,label='Nominal level'); axs[1,1].set_xticks([0,1],['Exchangeable null','Shifted alternative']); axs[1,1].set(ylabel='Empirical rate',title='Group-level conformal audit',ylim=(-.04,1.08)); axs[1,1].legend(frameon=False,loc='center right'); axs[1,1].grid(axis='y',alpha=.25)
for label,ax in zip('abcd',axs.ravel()): ax.text(-.12,1.08,label,transform=ax.transAxes,fontweight='bold',fontsize=13,va='top',ha='left')
fig.suptitle('Controlled evidence for the recommended P5 extension combination',fontsize=13,fontweight='bold')
for suffix in ('pdf','png'): fig.savefig(ROOT/'figures'/f'fig_stage66_extension_combination.{suffix}',dpi=300,bbox_inches='tight')
