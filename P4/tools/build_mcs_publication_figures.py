"""Vector publication layouts for MCS figures 1, 5 and 7.

Style follows the user's grouped workflow reference and the existing Stage 62
panel code. Only the visual grammar is reused; all content describes P4.
Numerical bar values are read unchanged from the frozen records.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/mcs_prediction_figures'
INK='#253849'
TEAL='#146a83'
GREY='#89999e'
GOLD='#ce9845'
NAMES=['ultraslow_two','mixed_four','uniform_slow','logwide_continuous']
LABELS=['Two-mode','Four-mode','Slow uniform','Log-wide']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    fig.canvas.draw()
    bounds=fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b=text.get_window_extent(fig.canvas.get_renderer())
            assert b.x0>=bounds.x0-1 and b.y0>=bounds.y0-1 and b.x1<=bounds.x1+1 and b.y1<=bounds.y1+1,(name,text.get_text(),b.bounds)
    fig.savefig(OUT/(name+'.pdf'),metadata={'CreationDate':None})
    fig.savefig(OUT/(name+'.png'),dpi=300)
    fig.savefig(OUT/(name+'.svg'))
    plt.close(fig)


def workflow():
    fig,ax=plt.subplots(figsize=(10.4,6.0))
    fig.subplots_adjust(left=.02,right=.98,bottom=.025,top=.98)
    ax.set(xlim=(0,12),ylim=(0,7.2));ax.axis('off')
    ax.add_patch(FancyBboxPatch((.13,6.57),11.74,.48,boxstyle='round,pad=.025,rounding_size=.1',fc=INK,ec=INK,lw=1))
    ax.text(6,6.81,'FREEZE THE TARGET, TOLERANCES AND INFORMATION CLASS',ha='center',va='center',fontsize=13.0,fontweight='bold',color='white')
    cards=[(.16,3.65,3.55,2.66),(4.24,3.65,3.55,2.66),(8.32,3.65,3.55,2.66),
           (8.32,.59,3.55,2.60),(4.24,.59,3.55,2.60),(.16,.59,3.55,2.60)]
    titles=['INFORMATION','FINITE RANGE','CONTINUOUS ENCLOSURE','TOLERANCE DECISION','AUDITABLE RECORD','INFORMATION IMPROVEMENT']
    colors=['#dceaf2','#e3eee5','#f4e9d3','#f4e2d9','#e7e3ef','#dcebe7']
    for number,(rect,title,color) in enumerate(zip(cards,titles,colors),1):
        x,y,w,h=rect
        ax.add_patch(FancyBboxPatch((x+.045,y-.04),w,h,boxstyle='round,pad=.02,rounding_size=.1',fc='#dfe3e6',ec='none',alpha=.6,zorder=0))
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.1',fc='white',ec=INK,lw=1.25,zorder=1))
        ax.add_patch(FancyBboxPatch((x,y+h-.52),w,.52,boxstyle='round,pad=.02,rounding_size=.09',fc=color,ec=INK,lw=1.0,zorder=2))
        ax.add_patch(Circle((x+.22,y+h-.26),.135,fc=INK,ec='none',zorder=3))
        ax.text(x+.22,y+h-.26,str(number),ha='center',va='center',fontsize=10,color='white',fontweight='bold',zorder=4)
        ax.text(x+w/2+.13,y+h-.26,title,ha='center',va='center',fontsize=10.6 if len(title)>20 else 11.8,color=INK,fontweight='bold',zorder=4)
    # Information icons show prescribed bands, not experimental observations.
    x,y,w,h=cards[0]
    ax.text(x+w/2,y+1.85,'Known loading and timing',ha='center',fontsize=11.3,color=INK)
    pts=np.array([.72,.62,.47,.38,.30])
    xx=np.linspace(x+.65,x+w-.50,len(pts)); yy=y+.95+pts*.65
    ax.plot(xx,yy,color=TEAL,lw=1.2)
    ax.errorbar(xx,yy,yerr=.10,fmt='o',color=TEAL,ms=3.5,capsize=3,lw=1.2)
    ax.text(x+w/2,y+.87,'Observation bands (schematic)',ha='center',fontsize=9.5,color='#536574')
    ax.text(x+w/2,y+.50,'Input envelope; declared priors',ha='center',fontsize=10.8,color=INK)
    ax.text(x+w/2,y+.22,'Check information assumptions',ha='center',fontsize=10.6,color=INK)
    x,y,w,h=cards[1]
    ax.text(x+w/2,y+1.84,'Uncapped positive class',ha='center',fontsize=11.3,color=INK)
    ax.text(x+w/2,y+1.36,r'$\kappa>\delta$',ha='center',fontsize=17,color=TEAL)
    ax.text(x+w/2,y+.99,'or',ha='center',fontsize=10.5,color=INK)
    ax.text(x+w/2,y+.59,r'$\kappa=\delta,\quad\gamma\geq\beta$',ha='center',fontsize=15,color=TEAL)
    ax.text(x+w/2,y+.19,'Qualified caps require their own class',ha='center',fontsize=9.0,color='#536574')
    x,y,w,h=cards[2]
    ax.text(x+w/2,y+1.85,'Physical inner / continuous outer',ha='center',fontsize=10.9,color=INK)
    positions=[x+.44,x+1.13,x+2.39,x+3.08]
    ax.plot(positions[::3],[y+1.25]*2,color=GREY,lw=9,solid_capstyle='round')
    ax.plot(positions[1:3],[y+1.25]*2,color=TEAL,lw=5,solid_capstyle='round')
    for xx,label in zip(positions,[r'$l_{\rm out}$',r'$l_{\rm in}$',r'$u_{\rm in}$',r'$u_{\rm out}$']):
        ax.plot([xx,xx],[y+1.09,y+1.39],color=INK,lw=1)
        ax.text(xx,y+.86,label,ha='center',fontsize=11,color=INK)
    ax.text(x+w/2,y+.49,'Bracket width and numerical error',ha='center',fontsize=10.7,color=INK)
    ax.text(x+w/2,y+.20,'Interval geometry is schematic',ha='center',fontsize=9.0,color='#536574')
    x,y,w,h=cards[3]
    ax.text(x+w/2,y+1.79,'Absolute-error rule shown',ha='center',fontsize=10.6,color='#536574')
    rows=[(1.30,r'$D_{\rm out}/2\leq\varepsilon$','Sufficient','#3f725a'),
          (.87,r'$D_{\rm in}/2>\varepsilon$','Insufficient','#a85849'),
          (.44,'Otherwise','Unresolved','#98733f')]
    for yy,rule,result,color in rows:
        ax.text(x+.18,y+yy,rule,va='center',fontsize=11.7,color=INK)
        ax.text(x+w-.16,y+yy,result,ha='right',va='center',fontsize=10.8,fontweight='bold',color=color)
    ax.text(x+w/2,y+.14,'Relative-error rule: Proposition 9',ha='center',fontsize=9.2,color='#536574')
    x,y,w,h=cards[4]
    for yy,line in zip([1.77,1.34,.91,.48],['Information class and target','Endpoint brackets','Decision and numerical status','Assumptions and provenance']):
        ax.text(x+.22,y+yy,'• '+line,va='center',fontsize=11.2,color=INK)
    x,y,w,h=cards[5]
    for yy,line in zip([1.77,1.34,.91,.48],['Loading / observation timing','Input calibration','Independently justified priors','Re-evaluate the target request']):
        ax.text(x+.20,y+yy,'• '+line,va='center',fontsize=10.9,color=INK)
    def arrow(a,b,color=INK,style='-|>',lw=1.65,**kw):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=20,lw=lw,color=color,zorder=6,**kw))
    arrow((3.77,4.91),(4.17,4.91))
    arrow((7.85,4.91),(8.25,4.91))
    arrow((8.75,3.58),(8.75,3.25))
    arrow((8.26,1.88),(7.85,1.88))
    arrow((4.44,3.57),(3.49,3.24),color='#a85849',connectionstyle='arc3,rad=.15')
    ax.text(5.0,3.37,'Unbounded',ha='center',va='center',fontsize=9,color='#a85849')
    arrow((11.4,3.26),(11.4,3.58),color=TEAL,linestyle='--',lw=1.3)
    ax.text(10.25,3.42,'Refine if unresolved',ha='center',fontsize=9.0,color=TEAL)
    # The information-improvement branch runs below the cards without crossing text.
    ax.plot([10.1,10.1,1.95],[.52,.27,.27],color='#a85849',lw=1.25,ls='--')
    arrow((1.95,.27),(1.95,.53),color='#a85849',lw=1.25)
    ax.text(6,.13,'Insufficient precision: improve information, then re-evaluate',ha='center',fontsize=9.3,color='#a85849')
    arrow((.36,3.24),(.36,3.58),color=TEAL,lw=1.3,linestyle='--')
    save(fig,'qualification_workflow')


def bars(study,comparator):
    geometry={'left':.08,'right':.985,'bottom':.27,'top':.86,'wspace':.30}
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.6))
    values5={}
    for ax,target,title in zip(axes,['G20','heldout_sigma20'],['Relaxation target G(20)','Held-out response at 20']):
        x=np.arange(4)
        for i,h in enumerate(['8.0','16.0']):
            values=[study['summaries'][n]['valid_mean_worst_diameters'][h][target] for n in NAMES]
            values5[target+'_'+h]=values
            ax.bar(x+(i-.5)*.34,values,.32,label='Window '+h[:-2],color=[GREY,TEAL][i],zorder=3)
        ax.axhline(.2 if target=='G20' else .1,color=INK,ls='--',lw=1,label='Frozen tolerance',zorder=2)
        ax.set_xticks(x,LABELS,rotation=15);ax.set_title(title,pad=9);ax.set_ylabel('Mean worst diameter')
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.17)
    fig.subplots_adjust(**geometry)
    handles,labels=axes[0].get_legend_handles_labels()
    legend=fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.015),ncol=3,frameon=False)
    fig.canvas.draw(); assert all(not legend.get_window_extent().overlaps(ax.get_window_extent()) for ax in axes)
    save(fig,'calibrated_practical_widths')
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.6))
    methods=['current_symmetric','classical_sign_convex','finite_grid_diagnostic']
    values7={}
    for ax,target,title in zip(axes,['G20','heldout_sigma20'],['Relaxation posterior','Held-out response posterior']):
        x=np.arange(4)
        for i,method in enumerate(methods):
            values=[comparator['summaries'][n]['methods'][method]['valid_mean_width'][target] for n in NAMES]
            values7[target+'_'+method]=values
            ax.bar(x+(i-1)*.24,values,.22,label=['Symmetric','Classical continuous','Finite grid diagnostic'][i],color=[GREY,TEAL,GOLD][i],zorder=3)
        ax.set_xticks(x,LABELS,rotation=15);ax.set_title(title,pad=9);ax.set_ylabel('Mean posterior width')
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.17)
    fig.subplots_adjust(**geometry)
    handles,labels=axes[0].get_legend_handles_labels()
    legend=fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.015),ncol=3,frameon=False)
    fig.canvas.draw(); assert all(not legend.get_window_extent().overlaps(ax.get_window_extent()) for ax in axes)
    save(fig,'classical_comparator')
    return {'figure_5':values5,'figure_7':values7}


def main():
    OUT.mkdir(parents=True,exist_ok=True);style()
    inputs=['results/p4_ultraslow_calibrated_validation_20261011.json','results/p4_matched_convex_baseline_20261011.json']
    study,comparator=[json.loads((ROOT/p).read_text(encoding='utf-8')) for p in inputs]
    workflow();values=bars(study,comparator)
    names=['qualification_workflow','calibrated_practical_widths','classical_comparator']
    report={'schema':'P4-MCS-publication-figures-v1','date':'2026-10-13',
            'input_sha256':{p:sha(ROOT/p) for p in inputs},'builder_sha256':sha(Path(__file__)),
            'generated_sha256':{n+'.'+ext:sha(OUT/(n+'.'+ext)) for n in names for ext in ['pdf','png','svg']},
            'saved_bar_values':values,'new_experiments':False,'all_numeric_values_from_frozen_records':True,
            'legends_outside_axes':True,'workflow_schematic':True,
            'reference_style_only':'User-supplied grouped workflow image and existing Stage 62 panel code; no P5 scientific content imported.'}
    (OUT/'publication_figures_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'figures':[1,5,7],'legends_outside_axes':True,'new_experiments':False}))


if __name__=='__main__':main()
