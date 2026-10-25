"""Integrate the append-only targeted evidence into the sole current revision."""
import argparse
from collections import Counter
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import zipfile

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P1=Path(__file__).resolve().parents[1]
REV=P1/'revision_neurocomputing_20261016'
RUN=REV/'targeted_validation_recovery_20261025'
CLOSE=REV/'targeted_validation_closure_20261025'
CURRENT=P1/'revise'
DOCS=P1/'docs/revise'
ZEN=P1/'zenodo_update_20261025'
STAGE=P1/'tmp/targeted_integration_20261025'
PAPER=STAGE/'manuscript'
BACKUP=P1/'archive/before_targeted_integration_20261025'
TAG='dfsc-revision-20261025'
RELEASE='https://github.com/hzhooning-art/DFSC/releases/tag/'+TAG
DOWNLOAD='https://github.com/hzhooning-art/DFSC/releases/download/'+TAG+'/DFSC_Revision_Reproducibility_20261025.zip'
DOI='10.5281/zenodo.23631530'


def sha(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def helper():
    path=REV/'archive/revise_delivery_20261022_before_visual_update_20261025/prepare_delivery.py'
    spec=importlib.util.spec_from_file_location('delivery_helper',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def data():
    rows=[read(p) for p in sorted((RUN/'results').glob('*/terminal/*.json'))]
    retry=read(CLOSE/'io_retry/terminal.json')
    effective=[retry if r['id']==retry['id'] else r for r in rows]
    return rows,effective,read(CLOSE/'AUDIT.json')


def figures():
    rows,effective,audit=data()
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    colors={'selective':'#087e8b','full_hp':'#b45535','original':'#6452a0','fdeint':'#b07812','pycaputo':'#3977ac'}
    labels={'selective':'Selective repair','full_hp':'All high precision','original':'Original hybrid',
            'fdeint':'FDEint','pycaputo':'pycaputo'}
    def save(fig,name):
        fig.savefig(PAPER/'figures'/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(STAGE/f'{name}.png',dpi=170,bbox_inches='tight')
        plt.close(fig)
    # Forward work/accuracy curves: all three alpha values and both problems retained.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.25),sharey=True)
    for ax,problem in zip(axes,['scalar','rotated2']):
        for method in ['fdeint','pycaputo']:
            for ai,alpha in enumerate([.5,.7,.9]):
                rr=sorted([r for r in rows if r.get('kind')=='forward' and r.get('problem')==problem
                           and r.get('method')==method and r.get('alpha')==alpha],key=lambda r:r['steps'])
                ax.plot([r['steps'] for r in rr],[r['relative_l2'] for r in rr],
                        color=colors[method],linestyle=['-', '--', ':'][ai],marker='o',ms=3,lw=1,
                        label=f'{labels[method]}, a={alpha:g}')
        for method in ['selective','full_hp','original']:
            rr=[r for r in rows if r.get('kind')=='forward' and r.get('problem')==problem and r.get('method')==method]
            ax.axhline(max(r['relative_l2'] for r in rr),color=colors[method],lw=1.3,
                       label=labels[method]+' (max over a)')
        ax.set(xscale='log',yscale='log',xlabel='Internal time steps (steppers only)',title=problem,
               xticks=[64,256,1024],xticklabels=['64','256','1024'])
        ax.grid(alpha=.2);ax.set_ylim(1e-17,.04)
    axes[0].set_ylabel('Relative trajectory L2 error')
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='lower center',bbox_to_anchor=(.5,-.19),ncol=3,frameon=False,fontsize=7)
    fig.subplots_adjust(bottom=.23,wspace=.12)
    save(fig,'fig_targeted_external')
    # All B denominators, including 8 out-of-contract probes, remain visible.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.25),gridspec_kw={'width_ratios':[1.1,1]})
    b=[r for r in rows if r['track']=='B']
    for label,marker,color,filter_ in [
        ('Uniform random (144)','o','#087e8b',lambda r:r['coordinate']['stratum']=='uniform'),
        ('Log-radius random (144)','.','#3977ac',lambda r:r['coordinate']['stratum']=='log_uniform'),
        ('Boundary probes (48)','x','#b45535',lambda r:r['coordinate']['stratum'] not in ('uniform','log_uniform'))]:
        rr=[r for r in b if filter_(r)]
        axes[0].scatter([r['coordinate']['alpha'] for r in rr],[max(r['coordinate']['radius'],1e-9) for r in rr],
                        s=13,marker=marker,c=color,label=label,alpha=.75)
    axes[0].set(xlabel='Fractional order a',ylabel='Radius |z| (nonpositive z)',yscale='log',title='Frozen scalar probes')
    axes[0].grid(alpha=.2)
    statuses=['accurate_return','inaccurate_return','refused','outside_contract_refused','outside_contract_returned']
    sc=['#087e8b','#b45535','#d9ac63','#afb6c0','#8e73b5']
    sl=['Accurate return','Inaccurate return','In-domain refusal','Outside: refused','Outside: returned']
    for i,method in enumerate(['guard','full_hp','original']):
        counts=Counter(next(x for x in r['methods'] if x['method']==method)['outcome'] for r in b)
        start=0
        for status,color in zip(statuses,sc):
            n=counts[status]
            axes[1].barh(i,n,left=start,color=color,height=.65)
            if n>=12:axes[1].text(start+n/2,i,str(n),ha='center',va='center',fontsize=8,color='white' if status in statuses[:2] else 'black')
            start+=n
    axes[1].set(yticks=range(3),yticklabels=['Guard only','High precision','Original hybrid'],xlabel='Coordinates (336 per method)',title='Returns and refusals')
    axes[1].invert_yaxis();axes[1].set_xlim(0,336)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(sc,sl)],loc='lower center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False,fontsize=7)
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.24),frameon=False,fontsize=7,ncol=1)
    fig.subplots_adjust(bottom=.26,wspace=.52)
    save(fig,'fig_targeted_stress')
    # Same six chosen operators, three optimizer initializations; no material replicates.
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.7),sharex=True,sharey=True)
    for row,regime in enumerate(['mild','hard']):
        for col,size in enumerate([4,8,16]):
            ax=axes[row,col]
            for method in ['selective','full_hp','original']:
                rr=[r for r in effective if r['track']=='C' and r['size']==size and r['regime']==regime and r['method']==method]
                yy=np.asarray([[x['loss'] for x in r['trajectory']] for r in rr])
                assert yy.shape==(3,100)
                ax.fill_between(range(1,101),yy.min(0),yy.max(0),color=colors[method],alpha=.08)
                ax.plot(range(1,101),np.median(yy,axis=0),color=colors[method],lw=1.2,
                        ls='--' if method=='full_hp' else '-',label=labels[method])
            ax.set_title(f'{regime}, {size} states');ax.set_yscale('log');ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('Training MSE')
            if row==1:ax.set_xlabel('Adam update')
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='lower center',bbox_to_anchor=(.5,-.01),ncol=3,frameon=False,fontsize=8)
    fig.subplots_adjust(bottom=.15,hspace=.35,wspace=.12)
    save(fig,'fig_targeted_training')
    # Bind every rendering input to the actual saved records and the explicit retry.
    put(STAGE/'plot_inputs.json',dict(records={str(p.relative_to(P1)):sha(p.read_bytes()) for p in (RUN/'results').glob('*/terminal/*.json')},
                                    retry=str((CLOSE/'io_retry/terminal.json').relative_to(P1)),retry_sha256=sha((CLOSE/'io_retry/terminal.json').read_bytes())))


FRAGMENT = r'''\subsection{Extended reference, solver and inverse-learning validation}
\label{sec:targeted-validation}
The preceding repair pilot motivates a separate frozen validation of new scalar
coordinates, external solver accuracy and longer inverse optimization. The
candidate wheel, numerical tolerances, coordinates and initializations are fixed
before this execution. The design follows the earlier results and is not blind
preregistration. A recovery run increases resource limits without tuning the
candidate; all earlier timeouts, partial trajectories and failed paths remain in
the artifact. Scalar records are reused byte-for-byte from the first execution
and are not counted as a second independent experiment. Table~\ref{tab:targeted-ledger}
separates attempted jobs from successful scientific outcomes.

\begin{table}[t]
\centering\small
\caption{Complete targeted-validation denominators. The explicit I/O retry is
one replacement analysis of the same task, not an additional replicate.}
\label{tab:targeted-ledger}
\begin{tabularx}{\linewidth}{@{}l r X@{}}
\toprule
Study & Planned & Recorded outcomes \\
\midrule
External/common problem & 175 & 78 forward returns, 54 gradient returns,
18 gradient-path errors, 24 completed inverse runs, 1 excluded API preflight \\
Scalar stress & 336 & 288 random coordinates and 48 boundary probes; all assessed \\
Joint inverse, original recovery & 72 & 62 completed 100-step runs, 9 refusals,
1 checkpoint I/O error after 64 updates \\
Joint inverse, explicit retry & 1 & Same failed task rerun from its frozen initialization;
100 updates completed and recovery criteria met \\
\bottomrule
\end{tabularx}
\end{table}

\paragraph{Previously untested scalar coordinates.}
A PCG64 generator with seed 20261023 samples 144 orders/radii uniformly and 144
with a log-uniform radius, within the scalar repair contract. All 288 are distinct
from the earlier 217 points. Another 48 probes inspect legacy branch neighborhoods
and contract edges; two boundary coordinates overlap earlier probes. Independent
defining-series references use 300 and 380 decimal digits, analytic first
sensitivities and agreement within $10^{-35}(1+|v|)$. Garrappa values and refined
finite differences provide separate checks; 16 preselected numerical inverse-Laplace
checks at 55/85 digits resolve and agree. The value target is
$10^{-12}+10^{-9}|v|$ and each first-sensitivity target is
$10^{-10}+10^{-9}|g|$.

The high-precision backend accurately returns all 288 new random cases and 38 of
40 in-domain boundary cases. It explicitly refuses two difficult edge cases at
$\alpha=0.45$ and $0.45000001$, $|z|=12$, under its finite work limits, and refuses
all eight out-of-contract probes. Thus 326 accurate returned coordinates, two
in-domain refusals and eight out-of-contract refusals are retained. The guard-only
path accurately returns 205 in-domain cases and refuses 123; outside its scoped
scalar-check entry, four of eight edge calls return, so that entry is not treated
as a whole-contract gate. The original hybrid has 102 inaccurate in-domain returns
under the same joint value/first-gradient test (Fig.~\ref{fig:targeted-stress}).
Agreement after float64 rounding is not exact-real zero error or a whole-domain
certificate. This new-coordinate evidence supports useful repair beyond the
failure-informed pilot, with the two remaining work-limit refusals explicit.

\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{fig_targeted_stress.pdf}
\caption{All 336 scalar coordinate cases. Left: frozen sampling strata; zero
and slightly negative out-of-domain radii are placed at $10^{-9}$ for the log-axis
display only, with exact coordinates retained in the artifact. Right: the full
return/refusal ledger for each method, including out-of-contract calls. No
application failure probability is inferred from this designed sample.}
\label{fig:targeted-stress}
\end{figure}

\paragraph{Matched external accuracy and gradient paths.}
All methods solve the same homogeneous Caputo system with either a scalar
eigenvalue 1 or a rotated two-state SPD operator with eigenvalues 0.3 and 6.
Orders are $0.5,0.7,0.9$ and six query times span $[0,1]$. FDEint and pycaputo
use 64, 128, 256, 512 and 1024 internal steps; direct spectral methods query the
same times without time stepping. The known homogeneous representation favors
direct propagation and does not replace a general forced/nonlinear solver test.
Both relative trajectory error and maximum absolute error must satisfy each
accuracy band. Figure~\ref{fig:targeted-external} retains all tested resolutions;
Table~\ref{tab:targeted-external} reports the complete method denominators.

\begin{table}[t]
\centering\small
\caption{Common-problem accuracy and gradient checks. Forward columns require
both relative and scaled absolute errors to pass. Gradient checks compare the
loss sensitivity in order and initial state to continuous references at
$10^{-7}+10^{-3}|g|$; these are not the scalar repair's tighter targets.
An own-discretization finite-difference audit is retained separately.}
\label{tab:targeted-external}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l r r r r@{}}
\toprule
Method & Forward cells & Pass $10^{-3}$ & Pass $10^{-4}$ & Gradient pass/checks \\
\midrule
Selective repair & 6 & 6 & 6 & 6/6 \\
All high precision & 6 & 6 & 6 & 6/6 \\
Original hybrid & 6 & 6 & 5 & 6/6 \\
FDEint & 30 & 7 & 0 & 2/18 \\
pycaputo & 30 & 27 & 21 & 13/18 \\
\bottomrule
\end{tabular*}
\end{table}

Selective repair's maximum relative forward error is $2.67\times10^{-13}$ over
its six cells. The physical-parameter FDEint path returns all 18 gradient cases;
its separately tested transformed upstream-order path raises autograd in-place
version errors in all 18 attempts. These errors are not silently repaired by
detaching and rewrapping a parameter, and do not imply that FDEint lacks all
order differentiation. pycaputo sensitivities here use explicit finite differences,
not an invented native autograd interface. Discrete-gradient consistency and
agreement with the continuous equation are distinct validation questions.

All 24 inverse tasks (two true orders, three initializations, four methods) complete
60 Adam updates and meet $|\widehat\alpha-\alpha_*|\leq0.02$ and independent
held-out relative trajectory error $\leq0.01$. FDEint uses its physical-parameter
path and pycaputo its explicit finite-difference loss gradient. Consequently,
successful parameter recovery alone does not distinguish DFSC from these controls.
The study records setup, including eigensystem construction, and end-to-end
optimization with explicit import/reference exclusions. Both the original A timing
and this recovery's shared-host timing are excluded from clean performance
rankings; no new runtime superiority claim follows.

\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{fig_targeted_external.pdf}
\caption{Forward accuracy at every tested stepper resolution and order. Horizontal
lines show each direct method's maximum error across the three orders, not a
stepper runtime. Both scalar and rotated two-state problems are retained.
The comparison measures accuracy on known homogeneous systems, with no
uncontended-cost claim.}
\label{fig:targeted-external}
\end{figure}

\paragraph{Larger, longer paired inverse trajectories.}
Six fixed QR-rotated SPD operators cover 4, 8 and 16 states in mild and difficult
spectral regimes. Each uses three optimizer initializations and four numerical
paths, giving 72 scheduled tasks. Two initial-state vectors and four training
times identify both order $\alpha$ and spatial exponent $\beta$; four disjoint
test times include $t=1.5$ beyond the training endpoint. Adam runs for 100 updates
with learning rate 0.05. The bounded parameterization is
$\alpha=0.45+0.5\sigma(r_1)$, $\beta=1+\sigma(r_2)$; the data-generating pairs
are $(0.72,1.4)$ and $(0.55,1.8)$. Recovery requires absolute error at most 0.02
in each parameter and held-out relative trajectory error at most 0.01.

The initial recovery execution completes 62 tasks, refuses nine difficult
guard-only tasks, and records one all-high-precision task interrupted by a
Windows checkpoint write error after update 64. A declared full rerun from the
same initialization completes 100 updates and meets recovery criteria. Including
that retry as the same task gives 63 completed/recovered tasks and nine refusals,
without adding a replicate or erasing the original error. All 18 selective versus
all-high-precision pairs pass the frozen trajectory comparison at updates
1, 12, 25, 50 and 100 (absolute tolerance $10^{-7}$, relative tolerance $10^{-6}$).
The largest checked discrepancy across parameters, gradients, losses and updates
is $4.99\times10^{-12}$. Figure~\ref{fig:targeted-training} shows every operator
and the range over its three initializations. The original hybrid also meets the
recovery criteria; the result supports numerical-fidelity preservation, not
universal prediction improvement. Six chosen operators and repeated starts are
not independent material samples or evidence of large-scale matrix-free,
GPU, forced or nonnormal repair.

\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{fig_targeted_training.pdf}
\caption{One hundred joint inverse updates on each fixed SPD operator.
Curves are median training MSE and shading spans the three optimizer
initializations, not a population confidence interval. Selective and full
high-precision curves nearly coincide. The difficult four-state full-precision
curve includes the disclosed same-task I/O retry; its original partial record
remains available. Guard-only hard tasks refuse and are not drawn as successful
zero-loss trajectories.}
\label{fig:targeted-training}
\end{figure}
'''


def prepare():
    assert not STAGE.exists() and not BACKUP.exists()
    BACKUP.mkdir(parents=True);PAPER.mkdir(parents=True)
    shutil.copytree(CURRENT,BACKUP/'revise');shutil.copy2(P1/'revise.zip',BACKUP/'revise.zip')
    shutil.copytree(DOCS,BACKUP/'docs')
    (BACKUP/'zenodo').mkdir()
    for path in ZEN.iterdir():
        if path.is_file():shutil.copy2(path,BACKUP/'zenodo'/path.name)
    with zipfile.ZipFile(CURRENT/'Manuscript_Source.zip') as archive:
        assert all((PAPER/n).resolve().is_relative_to(PAPER.resolve()) for n in archive.namelist())
        archive.extractall(PAPER)
    figures()
    (PAPER/'revision_targeted_validation.tex').write_text(FRAGMENT,encoding='utf-8')
    path=PAPER/'manuscript.tex';text=path.read_text(encoding='utf-8')
    text=text.replace(r'\input{revision_cpu_repair.tex}',r'\input{revision_cpu_repair.tex}'+'\n'+r'\input{revision_targeted_validation.tex}')
    text=text.replace('The optional repair backend returns values and first derivatives within the declared targets at 217 tested points, including the original failure cases. It passes 15 fixed-SPD action/loss-gradient checks and matches four short all-high-precision inverse-training trajectories at lower cost; the original hybrid remains faster.',
        'The optional repair backend corrects the original failure cases and accurately returns all 288 newly sampled scalar cases. Fixed-SPD validation extends to 16 states and 100 joint inverse updates, with 18 paired trajectories matching all-high-precision execution after one disclosed I/O retry. External solver comparisons separate trajectory accuracy, gradient paths and parameter recovery.')
    text=text.replace('A wider efficiency comparison would match achieved reference accuracy, device, queries and full setup/training cost across methods.',
        'Section~\\ref{sec:targeted-validation} adds reference-accuracy and gradient comparisons at multiple resolutions; clean isolated timing at matched accuracy remains needed for a wider efficiency ranking.')
    text=text.replace('Its inverse tasks are small and matched in structure, and the original hybrid attains slightly smaller final difficult-task prediction error.',
        'Its extended inverse tasks reach 16 states and 100 updates but remain small and matched in structure; the original hybrid also recovers the parameters and attains slightly smaller final prediction error in the earlier short pilot.')
    text=text.replace('The optional repair backend corrects the tested scalar failures, passes fixed-SPD action/gradient checks and follows the paired all-high-precision inverse trajectories at lower cost.',
        'The optional repair backend corrects the tested scalar failures and accurately returns all 288 new random scalar probes, with two explicit in-domain boundary refusals. Extended fixed-SPD tests produce 18 matching 100-update inverse trajectory pairs after a disclosed I/O retry; their shared-host timings do not support a new efficiency ranking.')
    # Keep reserved DOI honest while preparing an immutable public access route.
    old='Reviewer-accessible publication of the exact artifact remains to be verified before\nresubmission.'
    new=('The exact revision snapshot is prepared for the versioned GitHub release\\footnote{\\url{'+RELEASE+'}}. '
         'Publication and reviewer access remain pending separate verification before resubmission; the reserved Zenodo DOI does not yet identify a published deposit.')
    assert old in text
    text=text.replace(old,new)
    path.write_text(text,encoding='utf-8')
    print('Prepared three new figures, two tables, extended validation and conservative narrative updates.')


def supplementary_report():
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    import html
    rows,effective,audit=data();styles=getSampleStyleSheet();story=[]
    def para(text,style='BodyText'):
        story.append(Paragraph(html.escape(text),styles[style]))
        if style not in ('Heading1','Heading2','Title'):story.append(Spacer(1,7))
    def table(data,widths):
        rendered=[[Paragraph(str(cell),styles['BodyText']) for cell in row] for row in data]
        t=Table(rendered,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff0')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.grey),
            ('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7)]))
        story.append(t);story.append(Spacer(1,10))
    para('Targeted Validation: Reproduction and Complete Outcomes','Title')
    para('Supplement to NEUCOM-D-26-16369. Evidence snapshot: 25 October 2026. This report preserves failed, refused and retried outcomes; it is not a reply to reports that have not been received.')
    para('1. Fixed design and provenance','Heading2')
    para('The optional CPU repair wheel and numeric settings were unchanged. All 175 A jobs and 72 C jobs were freshly attempted in a resource-expanded recovery. All 336 B records were reused byte-for-byte, including adverse results. No fresh B timing or extra independent sample is claimed. Frozen protocols, coordinates, source/wheel hashes, raw attempts and process records are included in the accompanying archive.')
    para('Two freeze receipts changed only through document/plan relocation and manuscript archival. Exact original receipts matching both pre-execution hashes were retained. The append-only PROVENANCE_RECONCILIATION.json verifies every scientific binding digest and all original record bytes; neither original receipt nor terminal history was overwritten during this closeout.')
    table([['Track','Planned','Observed outcomes'],['A','175','78 forward returns; 54 gradient returns; 18 transformed-FDEint gradient errors; 24 recovered 60-update inversions; 1 excluded API preflight.'],
        ['B','336','288 new random cases and 48 boundary probes. All assessed; 326 accurate high-precision returns, 2 in-domain refusals and 8 out-of-domain refusals.'],
        ['C','72','Original recovery: 62 completed/recovered, 9 refused, 1 checkpoint PermissionError after 64 updates. One declared full same-task retry completed/recovered, giving 63 completed tasks and 18 complete matching selective/full-precision pairs.']],[35,45,410])
    para('2. External forward and gradient checks','Heading2')
    para('Scalar and rotated two-state homogeneous Caputo problems, orders 0.5/0.7/0.9, identical query times. Steppers use 64/128/256/512/1024 internal steps. The direct methods exploit the available exact spectral representation. Continuous-loss gradients and each implementation\'s own discrete finite-difference gradients are audited separately. pycaputo uses explicit finite differences, not a claimed native autograd path.')
    tab=[['Method','Forward n','Pass 1e-3','Pass 1e-4','Gradient pass/checks']]
    for method,stats in audit['a_method_statistics'].items():
        tab.append([method,stats['forward_cells'],stats['forward_pass_1e3'],stats['forward_pass_1e4'],f"{stats['continuous_gradient_pass']}/{stats['continuous_gradient_checks']}"])
    table(tab,[105,65,75,75,170])
    para('The FDEint physical-order path returns 18 gradient records; its separately tested transformed-order autograd path has 18 in-place-version errors. A finite-resolution continuous-gradient discrepancy is not automatically an incorrect derivative of the discrete solver. All 24 A inverse tasks recover their true order within 0.02 and held-out relative L2 within 0.01 after 60 Adam updates.')
    para('3. Scalar strata and complete method outcomes','Heading2')
    tab=[['Stratum','n','HP accurate/refused/outside','Original inaccurate (in-domain)']]
    for name,entry in audit['original_recovery']['b_strata'].items():
        hp=entry['methods']['full_hp']['outcomes'];orig=entry['methods']['original']['outcomes']
        tab.append([name,entry['planned'],f"{hp.get('accurate_return',0)} / {hp.get('refused',0)} / {hp.get('outside_contract_refused',0)}",orig.get('inaccurate_return',0)])
    table(tab,[100,35,185,170])
    para('New random coordinates are disjoint from all previous 217 scalar points. Two of the 48 boundary coordinates overlap previous probes. Independent references use 300/380 digits, analytic first sensitivities, separate Garrappa/finite-difference cross-checks and 16 resolved preselected inverse-Laplace checks. Targets are value atol 1e-12, gradient atol 1e-10 and rtol 1e-9. Float64 reference-rounding agreement is not exact-real zero error. Refused in-domain points are alpha=0.45 or 0.45000001, radius=12; no whole-domain certificate follows.')
    para('4. Joint inverse pairing and the declared I/O retry','Heading2')
    para('Six chosen fixed-SPD operators span sizes 4/8/16 and two regimes, each with three optimizer starts and four methods. Truth pairs are (0.72,1.4) and (0.55,1.8). Training times are 0.1/0.3/0.6/1.0; test times are 0.15/0.45/0.8/1.5. Adam takes 100 updates at rate 0.05. Both parameter errors must be at most 0.02 and held-out relative L2 at most 0.01.')
    tab=[['States','Regime','Completed after retry','Guard refusals','Complete matching pairs']]
    for size in [4,8,16]:
        for regime in ['mild','hard']:
            rr=[r for r in effective if r['track']=='C' and r['size']==size and r['regime']==regime]
            pairs=[q for q in audit['declared_io_retry_analysis']['c_pairs'] if q['size']==size and q['regime']==regime]
            tab.append([size,regime,sum(r['status']=='completed' for r in rr),sum(r['status']=='refused' for r in rr),sum(q['full_hundred_step_pair'] and q['status']=='matched_observed_steps' for q in pairs)])
    table(tab,[50,65,150,95,130])
    para('C-4-hard-307-full_hp originally failed only when replacing its checkpoint file after update 64. The original partial trajectory and error remain immutable. A separately declared full rerun uses the same candidate, settings, seed, initialization and 900-second limit. No checkpoint content was read while that worker ran. It completes 100 updates and recovers both parameters. It is the same task, not an extra replicate. All 18 selective/full-precision pairs pass the frozen step 1/12/25/50/100 tolerance (atol 1e-7, rtol 1e-6). Original-hybrid runs also recover, so no universal predictive advantage is claimed.')
    para('5. Cost qualification, testing and reproduction','Heading2')
    para('Basis/model/optimizer setup and end-to-end loop times are saved with explicit exclusions. The original A run had known CPU contention and the recovery used a shared host. Both sets of timings remain raw observations, excluded from clean performance rankings. Existing isolated/earlier finite cost results retain their original narrower interpretation.')
    para('Final regression: 47 targeted tests + 55 core evidence tests + 10 scalar tests + 9 spectral tests + 6 delivery tests = 127 passed. A restricted-sandbox attempt timed out and is retained; the same suites passed with required Windows process/temp-file access. No new independent material samples, GPU repair, general nonnormal solver, higher derivatives or global floating-point guarantee are added.')
    para('Archive reproduction: start with README.md and REPRODUCE.md; numerical snapshots retain their recorded dependency versions. Run the included verify_targeted_evidence.py to check every frozen binding, raw record, retry and pairing without re-running a training study. Full scientific regeneration is substantially more costly than an evidence replay. Main figures are reproducible with tools/integrate_targeted_revision_20261025.py and the retained plot-input hashes.')
    for name,title in [('fig_targeted_stress','All scalar probes and outcomes'),('fig_targeted_external','All external forward resolutions'),('fig_targeted_training','All joint-inverse operators')]:
        story.append(PageBreak());para(title,'Heading1')
        img=Image(str(STAGE/(name+'.png')));img.drawHeight*=490/img.drawWidth;img.drawWidth=490;story.append(img)
    target=STAGE/'Targeted_Validation_Report.pdf'
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.drawString(45,24,'DFSC | Targeted validation | 25 October 2026');canvas.drawRightString(550,24,str(doc.page))
    SimpleDocTemplate(str(target),pagesize=(595.276,841.89),leftMargin=45,rightMargin=45,topMargin=40,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)


def editorial():
    labels={n:dict(number=num,page=int(page)) for n,num,page in re.findall(
        r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}',(PAPER/'manuscript.aux').read_text(encoding='utf-8'))}
    put(STAGE/'page_label_map.json',labels)
    section=labels['sec:targeted-validation']['number']
    response=(BACKUP/'revise/submission/Response_to_AE_Draft.md').read_text(encoding='utf-8')
    response=response.replace('The 34-page visual revision changes pagination; see COMMENT_CLOSURE_MATRIX.md for current page numbers.',
        'The extended revision changes pagination; current section, figure, table and page locations are supplied in COMMENT_CLOSURE_MATRIX.md.')
    insert={
      'AE-1':f'Section {section}, Tables 17-18 and Figure 15 add the complete same-problem multi-resolution comparison: 78 forward cells, 72 gradient/path attempts and 24 inverse runs. Direct spectral methods use the known homogeneous representation; FDEint/pycaputo are history-aware steppers. Forward pass counts at 1e-4 are 6/6 selective, 6/6 all-high-precision, 5/6 original, 0/30 FDEint and 21/30 pycaputo, with every tested resolution retained. All 24 inverse runs recover, so we do not infer universal predictive superiority. Current shared-host timings are disclosed but excluded from clean performance rankings.',
      'AE-2':f'Section {section} and Figure 14 extend the failure-informed pilot with 288 new random scalar points, all disjoint from the previous 217. The unchanged repair candidate accurately returns all 288 and 38/40 in-domain boundary cases; two work-limit refusals and all eight out-of-domain refusals remain explicit. Independent 300/380-digit series/analytic-sensitivity references, separate value/finite-difference checks and 16 preselected inverse-Laplace checks support these finite conclusions. The 336 records are reused byte-for-byte from their first execution, not counted twice. No strict floating-point enclosure or global accuracy theorem is claimed.',
      'AE-3':f'The added common-problem audit in Section {section} distinguishes native/explicit gradients, own-discretization finite differences and continuous-reference gradients. Selective and full high precision pass all six continuous-gradient cases each under the disclosed loss-gradient targets. FDEint physical-order returns 18 cases, while its separate transformed-order path has 18 recorded autograd in-place-version errors; no detach/rewrap hides these failures. pycaputo uses explicit finite differences. Figure 16 extends paired training to six chosen 4/8/16-state operators, three initializations and 100 updates; all 18 selective/full-precision pairs match the frozen comparison tolerances after one declared same-task I/O retry.',
      'AE-4':f'The extended study in Section {section} records basis/model/optimizer setup and complete optimization with explicit reference/import exclusions. The first A timing run was contaminated by a residual CPU task; the new recovery used a shared host. We retain both histories and exclude both from clean runtime rankings. The longer training supports numerical fidelity and successful recovery, not a new uncontended performance claim. Earlier setup-inclusive measurements retain their finite-task scope.',
      'AE-7':'The revised reproducibility archive now includes both original and recovery histories, the exact frozen source/wheel candidate, all 583 recovery terminal records, the separate I/O retry, receipt-relocation reconciliation, final 127-test regression records, plot sources and a standalone evidence verifier. The planned GitHub release route is '+RELEASE+'. Public upload and independent download/hash verification remain pending; the Zenodo DOI remains reserved and is not represented as published. Source, marked/clean paper and the accessible archive must have matching hashes before the response is finalized.'}
    for key,paragraph in insert.items():
        match=re.search(r'^## '+key+r'\..*?(?=^## |\Z)',response,re.M|re.S);assert match
        block=match.group(0).rstrip()+'\n\nAdditional targeted validation (25 October 2026): '+paragraph+'\n\n'
        response=response[:match.start()]+block+response[match.end():]
    response=response.replace('Authors must supply the final artifact access information and confirm that reviewers\ncan reach it. This paragraph cannot be treated as final public-access confirmation.',
        'The exact archive is locally prepared, but publication and unauthenticated download/hash verification remain pending. This draft still requires complete reviewer reports and author review before portal submission.')
    response=response.replace('Its files have not yet been uploaded or published, and reviewer access remains unverified.',
        'The Zenodo files have not been published. A separate exact-version GitHub route is prepared; publication and access verification remain pending.')
    (STAGE/'Response_to_AE_Draft.md').write_text(response,encoding='utf-8')
    cover=(BACKUP/'revise/submission/Cover_Letter_Draft.md').read_text(encoding='utf-8')
    cover=cover.replace('New finite scalar/action/gradient checks and short\ninverse-training comparisons support actual bounded repair rather than refusal alone.',
        'New scalar/action/gradient checks, external solver comparisons and joint inverse studies up to 16 states and 100 updates support actual bounded repair rather than refusal alone. All 288 new random scalar probes return accurately; all 18 complete selective/full-high-precision training pairs match after one disclosed checkpoint-I/O retry.')
    cover=cover.replace('Its files have not yet been uploaded or published, and reviewer access remains unverified.',
        'Its Zenodo files have not yet been published. The exact revision artifact is prepared for the versioned GitHub route '+RELEASE+'. Public upload and independent download/hash verification remain pending.')
    cover=cover.replace('confirmation of\nthe exact reviewer-accessible artifact route and final portal checks.', 'confirmation of final artifact alignment and final portal checks.')
    cover=cover.replace('external release or submission is asserted.', 'journal submission is asserted.')
    (STAGE/'Cover_Letter_Draft.md').write_text(cover,encoding='utf-8')
    # PDF response generation with a current footer (the old helper has a fixed old date).
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
    import html
    for stem in ['Response_to_AE_Draft','Cover_Letter_Draft']:
        styles=getSampleStyleSheet();body=ParagraphStyle('body',parent=styles['BodyText'],fontSize=10,leading=14,spaceAfter=8,splitLongWords=True)
        heading=ParagraphStyle('heading',parent=body,fontName='Helvetica-Bold',fontSize=12,leading=16,spaceBefore=10,keepWithNext=True)
        story=[]
        for block in re.split(r'\n\s*\n',(STAGE/(stem+'.md')).read_text(encoding='utf-8').strip()):
            line=' '.join(block.splitlines());is_heading=line.startswith('#')
            line=html.escape(re.sub(r'^#{1,3}\s+','',line));line=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',line)
            story.append(Paragraph(line,heading if is_heading else body))
        def footer(canvas,doc):
            canvas.setFont('Helvetica',8);canvas.drawString(45,24,'NEUCOM-D-26-16369 | Author-review draft | 25 October 2026');canvas.drawRightString(550,24,str(doc.page))
        SimpleDocTemplate(str(STAGE/(stem+'.pdf')),pagesize=(595.276,841.89),leftMargin=45,rightMargin=45,topMargin=42,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
    supplementary_report()
    print('Prepared AE response additions, current cover letter and complete supplementary report.')


def finalize():
    document=fitz.open(PAPER/'manuscript.pdf');pages=len(document)
    text=(PAPER/'manuscript.tex').read_text(encoding='utf-8')
    expanded=re.sub(r'\\input\{([^}]+)\}',lambda m:(PAPER/m.group(1)).read_text(encoding='utf-8'),text)
    counts=dict(pages=pages,figures=len(re.findall(r'\\includegraphics',expanded)),
                tables=len(re.findall(r'\\begin\{table\}',expanded)),
                references=len(re.findall(r'\\bibitem',(PAPER/'manuscript.bbl').read_text(encoding='utf-8'))))
    assert counts==dict(pages=38,figures=16,tables=18,references=36),counts
    assert not re.search(r'undefined|Overfull|multiply defined',(PAPER/'manuscript.log').read_text(encoding='utf-8'),re.I)
    with zipfile.ZipFile(BACKUP/'revise/Manuscript_Source.zip') as previous:
        for n in previous.namelist():
            if n.startswith('figures/'):assert previous.read(n)==(PAPER/n).read_bytes(),n
    shutil.copy2(PAPER/'manuscript.pdf',CURRENT/'Manuscript.pdf')
    with zipfile.ZipFile(CURRENT/'Manuscript_Source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(PAPER.rglob('*')):
            if path.is_file() and (path.suffix in ['.tex','.bib','.bbl'] or path.parent.name=='figures'):
                z.write(path,path.relative_to(PAPER).as_posix())
    marking=helper().marked_pdf(CURRENT/'Manuscript.pdf',P1/'submission_neurocomputing/manuscript.pdf',STAGE/'Manuscript_Marked.pdf')
    shutil.copy2(STAGE/'Manuscript_Marked.pdf',CURRENT/'submission/Manuscript_Marked.pdf')
    for name in ['Response_to_AE_Draft.md','Response_to_AE_Draft.pdf','Cover_Letter_Draft.md','Cover_Letter_Draft.pdf','Targeted_Validation_Report.pdf']:
        shutil.copy2(STAGE/name,CURRENT/'submission'/name)
    abstract=' '.join(text.split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0].replace('--','-').split())
    field=CURRENT/'submission/Portal_Title_Abstract_Keywords.txt'
    fields=field.read_text(encoding='utf-8');a=fields.index('ABSTRACT\n')+len('ABSTRACT\n');b=fields.index('\n\nKEYWORDS',a)
    field.write_text(fields[:a]+abstract+fields[b:],encoding='utf-8')
    highlights=['DFSC embeds trainable Mittag-Leffler propagation in PyTorch learning graphs.',
        'All 288 new random scalar probes pass joint value and first-gradient checks.',
        'Selective repair matches 18 full-precision 100-update inverse trajectories.',
        'External comparisons distinguish forward accuracy, gradients and recovery.',
        'Frozen records retain refusals, adverse results and a disclosed I/O retry.']
    assert all(len(x)<=85 for x in highlights)
    (CURRENT/'submission/Highlights.txt').write_text('\n'.join('- '+x for x in highlights)+'\n',encoding='utf-8')
    (CURRENT/'submission/Data_and_Code_Access_Draft.txt').write_text(
        'Historical public dfsc 0.1.0: https://doi.org/10.5281/zenodo.21588834. '
        'The exact revision artifact (guard-only and optional CPU repair sources/wheels, protocols, complete targeted histories and saved results) has a separately versioned GitHub route: '+RELEASE+'. '
        'Planned direct archive: '+DOWNLOAD+'. Public upload and external download/hash verification remain pending. '
        'The reserved Zenodo DOI '+DOI+' has not been published and is not a substitute for that accessible snapshot. '
        'See README.md, REPRODUCE.md and ARCHIVE_MANIFEST.json in the archive.\n',encoding='utf-8')
    metadata=CURRENT/'submission/Submission_Metadata.md'
    s=metadata.read_text(encoding='utf-8')
    s=s.replace('Code availability: historical public release/DOI distinct from local candidates; current reviewer access pending.',
        'Code availability: historical public release/DOI distinct from revision candidates; planned exact-version GitHub route '+RELEASE+'; public upload and external verification pending.')
    s+='\n## Extended evidence snapshot (25 October 2026)\n\n'
    s+='Current paper: 38 pages, 16 figures, 18 tables, 36 references. Targeted_Validation_Report.pdf is the optional readable supplement; raw data/software are in the exact-version artifact.\n'
    s+='Full Reviewer 1/3 reports, all approved biographies/photos and portal confirmations remain missing (author confirmed 25 October). No final submission has been made.\n'
    metadata.write_text(s,encoding='utf-8')
    (CURRENT/'README.md').write_text(f'''# DFSC revision: sole current delivery

Current clean manuscript: {pages} pages, 16 figures, 18 tables and 36 references.
All 13 previous figures and all previous data remain; the extended validation adds
three figures, two tables and a complete readable outcome report.

- Manuscript.pdf: current clean paper.
- Manuscript_Source.zip: exact LaTeX, bibliography and all 16 vector figures.
- submission/Manuscript_Marked.pdf: same text, added/replaced words highlighted
  against the local original; it is not native LaTeX tracked deletion markup.
- submission/Response_to_AE_Draft.pdf and .md: nine local AE-summary items,
  updated with new evidence; complete reviewer reports remain unavailable.
- submission/Targeted_Validation_Report.pdf: readable supplementary evidence.
- submission/: cover draft, highlights, author/portal and data/code fields.

The whole current submission packet is ../revise.zip. Numerical software and raw
records are separately packaged; do not upload the old archival manuscript.
Exact artifact route: {RELEASE}
Consult ../docs/revise/PUBLIC_ACCESS_VERIFICATION.json for actual external status.
Zenodo DOI {DOI} is reserved, not published. The staged Zenodo ZIP contains
the same clean paper and exact source as this delivery.

Still required: full reports, author-approved biographies/photos, author review,
portal ORCIDs/online responses and merged-PDF approval. No journal submission
or approval of missing materials is inferred from this packet.
''',encoding='utf-8')
    labels=read(STAGE/'page_label_map.json');put(DOCS/'page_label_map.json',labels);put(DOCS/'marking.json',marking)
    matrix=read(BACKUP/'docs/closure_matrix.json')
    mapping={'AE-1':['sec:targeted-validation','tab:targeted-external','fig:targeted-external'],
             'AE-2':['sec:targeted-validation','fig:targeted-stress'],
             'AE-3':['sec:targeted-validation','fig:targeted-training'],
             'AE-4':['sec:targeted-validation'], 'AE-7':['sec:targeted-validation']}
    remaining={'AE-1':'Executed accuracy/gradient comparisons for two steppers; other ecosystems are inspected. New shared-host costs are not clean performance evidence.',
               'AE-2':'288 new random points and explicit boundary refusals support finite empirical reliability; no global floating-point certificate.',
               'AE-3':'18 complete fixed-SPD training pairs after a disclosed I/O retry; no global switching smoothness or higher-order derivative guarantee.',
               'AE-4':'Existing setup-inclusive costs retained. New shared-host records cannot establish an uncontended ranking.',
               'AE-7':'127 final regression tests passed; exact archive prepared with integrity replay. Public access must match actual download verification; Zenodo DOI reserved.'}
    lines=['# AE-summary response and evidence map','',
        'These local identifiers paraphrase the supplied AE summary. Full reviewer reports are missing; this is not a claim that every reviewer comment is closed.','',
        '| Item | Current locations | Remaining scope |','|---|---|---|']
    for row in matrix['items']:
        if row['id'] in mapping:
            row['labels']=list(dict.fromkeys(row['labels']+mapping[row['id']]))
            row['remaining']=remaining[row['id']]
            row['targeted_evidence']='targeted_validation_closure_20261025/AUDIT.json'
        row['locations']=[dict(label=l,**labels[l]) for l in row['labels'] if l in labels]
        loc='; '.join(f"{x['label']} {x['number']} (p.{x['page']})" for x in row['locations']) or 'Title / References'
        lines.append(f"| {row['id']} | {loc} | {row['remaining']} |")
    matrix['date']='2026-10-25';matrix['targeted_validation_integrated']=True
    put(DOCS/'closure_matrix.json',matrix)
    (DOCS/'COMMENT_CLOSURE_MATRIX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (DOCS/'MARKING_METHOD.md').write_text(f'''# Current marked-copy method

Clean paper: {pages} pages; SHA-256 {sha((CURRENT/'Manuscript.pdf').read_bytes())}.
Yellow printable annotations mark added/replaced normalized words against the
local original submission PDF. This preserves clean-paper text and pagination;
it does not display deleted words or prove editor approval of the format.
All old figures are retained; three new figures use the bound raw records and
the declared same-task I/O retry. See SOURCE_CHANGES.patch for source differences.
''',encoding='utf-8')
    original=(P1/'submission_neurocomputing/manuscript.tex').read_text(encoding='utf-8')
    (DOCS/'SOURCE_CHANGES.patch').write_text(''.join(difflib.unified_diff(original.splitlines(keepends=True),expanded.splitlines(keepends=True),fromfile='local-original/manuscript.tex',tofile='current-expanded/manuscript.tex')),encoding='utf-8')
    gate=read(BACKUP/'docs/GATE.json')
    gate.update(targeted_validation_integrated=True,targeted_all_jobs_attempted=True,
        targeted_same_task_io_retry_success=True,targeted_complete_training_pairs=18,
        targeted_regression_tests_passed=127,targeted_freeze_relocation_reconciled=True,
        new_experiments=True,mathematical_content_and_results_unchanged=False,
        mathematical_theorems_unchanged=True,prior_figures_and_data_retained=True,
        narrative_minor_revision=False,reviewer_replies_preserved_byte_for_byte=False,
        clean_manuscript_science_unchanged=False,scientific_results_and_figures_unchanged=False,
        clean_manuscript_prepared=True,source_and_all_16_figures_prepared=True,
        full_reports_received=False,approved_bios_photos_received=False,resubmission_ready=False,
        targeted_uncontended_runtime_claim=False,full_reports_and_author_materials_confirmed_missing='2026-10-25')
    put(DOCS/'GATE.json',gate)
    remaining_file=DOCS/'AUTHOR_ONLY_REMAINING.md'
    remaining_text=(BACKUP/'docs/AUTHOR_ONLY_REMAINING.md').read_text(encoding='utf-8').replace('日期：2026-10-22','日期：2026-10-25')
    remaining_text+='\n## 本轮核实\n\n作者于2026-10-25确认仍无完整审稿报告、作者批准的简介和照片。本轮未虚构补齐。定向实验已收尾并纳入稿件，原始 I/O 失败和重跑均保留；公共访问以 PUBLIC_ACCESS_VERIFICATION.json 实际核验为准。\n'
    remaining_file.write_text(remaining_text,encoding='utf-8')
    put(DOCS/'targeted_integration.json',dict(**counts,old_figures_retained=13,new_figures=3,new_tables=2,
        abstract_words=len(abstract.split()),manuscript_sha256=sha((CURRENT/'Manuscript.pdf').read_bytes()),
        source_sha256=sha((CURRENT/'Manuscript_Source.zip').read_bytes()),
        targeted_audit_sha256=sha((CLOSE/'AUDIT.json').read_bytes()),io_retry_disclosed=True,
        full_reviewer_reports_available=False,author_bios_photos_available=False))
    # Stable top-level status, before external access is independently checked.
    (P1/'CURRENT_STATUS.md').write_text(f'''# P1 当前状态

更新：2026-10-25。Neurocomputing 大修，NEUCOM-D-26-16369。
决定信截止：2026-11-02 23:59，门户时区待确认；尚未最终提交。

唯一当前正文：revise/Manuscript.pdf；源码：revise/Manuscript_Source.zip。
当前正文 {pages} 页、16 图、18 表、36 参考文献；保留既有13图，新增3图2表。
同版标记稿、AE回复草稿、说明信、可读补充报告和上传字段在 revise/submission/。
当前完整投稿包：revise.zip。本轮已覆盖更新，旧版仅在 archive/ 保存。

定向验证：A175项全部尝试，24项60步反演恢复；B336记录原样复用，288新增随机点均准确返回；
C72项全部尝试，原始62完成/9拒绝/1第64步I/O错误，声明后的同任务重跑成功；
有效任务计数为63完成恢复/9拒绝，18组完整100步配对一致。失败记录未被替换或删除。
冻结收据仅路径迁移的冲突已凭精确原收据核对；127项最终回归通过。
共享主机计时不得用作无争用性能排名；本轮没有增加独立材料样本或全局误差保证。

最新版 Zenodo 包：zenodo_update_20261025/DFSC_P1_Zenodo_Update_20261025.zip。
新 DOI {DOI} 仅预留，公开API返回404，未确认发布。
GitHub工件路线：{RELEASE}；实际上传/下载状态见 docs/revise/PUBLIC_ACCESS_VERIFICATION.json。

作者确认仍缺完整 Reviewer 1/3 报告及四位作者批准的简介、照片。
本地三/四作ORCID正确；门户ORCID、在线回复、全体作者确认和合并PDF审核仍需完成。
回复覆盖已知AE总结，不能据此声称完整审稿意见全部关闭。
''',encoding='utf-8')
    reseal_revise()
    print(json.dumps(counts))


def reseal_revise():
    exclusions={'MANIFEST_SHA256.json','SHA256SUMS.txt'}
    hashes={p.relative_to(CURRENT).as_posix():sha(p.read_bytes()) for p in sorted(CURRENT.rglob('*')) if p.is_file() and p.name not in exclusions}
    (CURRENT/'SHA256SUMS.txt').write_text(''.join(v+'  '+k+'\n' for k,v in hashes.items()),encoding='utf-8')
    hashes['SHA256SUMS.txt']=sha((CURRENT/'SHA256SUMS.txt').read_bytes())
    put(CURRENT/'MANIFEST_SHA256.json',hashes)
    with zipfile.ZipFile(P1/'revise.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(CURRENT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(CURRENT).as_posix())


def package():
    # Curated previous archive retains all prior science/licenses, with current paper replaced.
    previous=BACKUP/'zenodo/DFSC_P1_Zenodo_Update_20261025.zip'
    with zipfile.ZipFile(previous) as z:payload={n:z.read(n) for n in z.namelist()}
    payload['Manuscript.pdf']=(CURRENT/'Manuscript.pdf').read_bytes()
    payload['Manuscript_Source.zip']=(CURRENT/'Manuscript_Source.zip').read_bytes()
    with zipfile.ZipFile(CURRENT/'Manuscript_Source.zip') as z:
        for n in z.namelist():payload['manuscript/'+n]=z.read(n)
    payload['Targeted_Validation_Report.pdf']=(CURRENT/'submission/Targeted_Validation_Report.pdf').read_bytes()
    paths=[]
    for root in [REV/'targeted_validation_20261023',RUN,CLOSE]:
        paths.extend(p for p in root.rglob('*') if p.is_file() and
            not any(part in ('__pycache__','tmp','output') for part in p.relative_to(root).parts) and p.name!='execution.lock')
    for root in [REV/'targeted_validation_20261023',RUN]:
        for binding in read(root/'freeze_receipt.json')['bindings'].values():
            path=(root/binding['path']).resolve()
            assert sha(path.read_bytes())==binding['sha256']
            paths.append(path)
    for proof in read(CLOSE/'PROVENANCE_RECONCILIATION.json')['proofs']:paths.append(P1/proof['original_receipt'])
    for rel in ['close_targeted_validation_20261025.py','integrate_targeted_revision_20261025.py','verify_targeted_evidence.py']:
        paths.append(P1/'tools'/rel)
    paths.extend([STAGE/'plot_inputs.json'])
    for path in set(paths):
        name=path.relative_to(P1.parent).as_posix()
        payload[name]=path.read_bytes()
    for n in ['P1/tools/verify_targeted_evidence.py']:
        assert n in payload
    metadata=json.loads(payload['.zenodo.json'])
    metadata['title']='DFSC: Differentiable Mittag-Leffler Propagation and Numerical Admission for Fractional Scientific Machine Learning - Revision Reproducibility Artifacts'
    metadata['description']=(
        'Reproducibility snapshot accompanying the revised DFSC manuscript. Includes the current 38-page paper, exact LaTeX and 16 vector figures, historical software, separately namespaced optional CPU repair wheel/source, frozen protocols, all original and recovery outcomes, and reproduction scripts. '
        'The targeted evidence comprises 175 external/composition jobs, 336 scalar-coordinate records reused without duplication, and 72 joint inverse tasks up to 16 states and 100 updates. All 288 new random scalar points return accurately; two in-domain edge refusals and eight outside-domain refusals remain explicit. '
        'A declared rerun of one checkpoint-I/O failure completes the original task, yielding 18 full selective/all-high-precision trajectory pairs. Original errors, refusals and earlier partial runs are preserved. Final regression suites pass 127 tests; the standard-library verifier checks frozen bindings, complete records and paired trajectories without retraining. '
        'Shared-host timings support no uncontended performance ranking. The evidence does not establish global floating-point certification, general GPU/nonnormal/forcing repair, independent-material transfer or universal predictive superiority. The artifact version is a research snapshot; software distribution versions are unchanged. Third-party data retain their licenses. '
        'Reserved Zenodo DOI: '+DOI+'. GitHub exact-version route: '+RELEASE+'.')
    metadata['related_identifiers']=[dict(identifier=RELEASE,relation='isIdenticalTo',scheme='url')]
    def js(n,obj):payload[n]=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
    js('.zenodo.json',metadata)
    js('PUBLICATION_STATUS.json',dict(doi=DOI,status='reserved_draft',zenodo_uploaded=False,zenodo_published=False,
        zenodo_public_api_status=404,github_release_route=RELEASE,
        github_upload_verified_at_build=False,external_verification_receipt_separate=True))
    provenance=json.loads(payload['provenance/input_archives.json'])
    provenance.update(current_manuscript_sha256=sha(payload['Manuscript.pdf']),
        current_revise_zip_sha256=sha((P1/'revise.zip').read_bytes()),targeted_validation_integrated=True,
        original_and_retry_histories_retained=True,all_old_figures_retained=True,
        new_scientific_experiments=True,github_release=RELEASE)
    # Replace stale bindings for every retained member, and bind new scientific members.
    for n in list(provenance.get('retained_file_bindings',{})):
        if n in payload:provenance['retained_file_bindings'][n]['sha256']=sha(payload[n])
    provenance['targeted_members']={p.relative_to(P1.parent).as_posix():sha(p.read_bytes()) for p in set(paths)}
    js('provenance/input_archives.json',provenance)
    payload['README.md']=f'''# DFSC revision reproducibility artifacts

Snapshot 0.1.0-revision.20261025. Current paper: 38 pages, 16 figures, 18 tables,
36 references. Exact paper/source are Manuscript.pdf, Manuscript_Source.zip and
manuscript/. No reviewer correspondence, cover letter, biographies or credentials
are part of this public archive. Software distribution versions are unchanged.

Targeted_Validation_Report.pdf explains the full denominators, adverse outcomes,
same-task I/O retry, accuracy/gradient targets and cost qualification.
Start with REPRODUCE.md. All 583 recovery terminal records and original histories
are retained under P1/revision_neurocomputing_20261016/; B is reused, not duplicated
as new evidence. The closure directory contains the exact receipt-relocation
proof, explicit I/O retry, 127-test final regression and complete audit.

Portable read-only evidence replay (standard Python; no model training):
    python -B P1/tools/verify_targeted_evidence.py

The original Windows freeze receipts retain exact bytes and binding hashes.
The portable verifier normalizes path separators. Frozen reference/solver modules
are supplied for scientific reproduction with their recorded dependency versions;
raw timing observations are not qualified uncontended-performance evidence.
The optional backend's source and wheel remain in cpu_repair_delivery_20261021/.

Historical dfsc: 0.1.0. Guard-only candidate: 0.1.1.dev0. Optional CPU repair:
dfsc-cpu-repair-candidate 0.1.0.dev0, separate dfsc_cpu_repair namespace.
No global error certificate, general repair backend, new physical replicates or
universal prediction advantage is asserted. Third-party licenses remain intact.

Reserved Zenodo DOI: {DOI} (not yet published).
Exact-version GitHub route: {RELEASE}
Publication/download is verified outside this immutable snapshot after upload.
'''.encode()
    payload['REPRODUCE.md']+=b'''\n\n## Extended targeted evidence (25 October 2026)\n\nRun `python -B P1/tools/verify_targeted_evidence.py` from the extracted archive.
This checks the actual receipt digests, every frozen content binding, all 583
terminal identities, B byte reuse, the separately declared same-task I/O retry,
18 complete trajectory pairs and recorded 127-test regression outcomes. It does
not rerun the experiments or claim a fresh scientific-dependency installation.
The original shared-host controller remains marked needs_attention: it stopped
on receipt-path changes. Closure is append-only in targeted_validation_closure_20261025;
its proof reconciles that historical status without falsifying or overwriting it.
The original I/O error is retained. Use the closure AUDIT's explicitly labelled
retry analysis, not an unexplained substitution or an extra independent replicate.
'''
    payload['CHANGELOG.md']+=b'\n2026-10-25 targeted integration: 288 new random scalar probes; complete A/C attempt; one explicit checkpoint-I/O retry; 18 full training pairs; 127 final tests; 3 new figures and 2 tables. Original adverse evidence retained.\n'
    js('ARCHIVE_MANIFEST.json',dict(version=metadata['version'],files={n:sha(b) for n,b in sorted(payload.items()) if n!='ARCHIVE_MANIFEST.json'}))
    name='DFSC_P1_Zenodo_Update_20261025.zip'
    with zipfile.ZipFile(ZEN/name,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n,b in sorted(payload.items()):z.writestr(n,b)
    # Synchronize the prepared directory as well as the upload ZIP.
    for n,b in payload.items():
        target=ZEN/'files'/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
    put(ZEN/'zenodo_metadata.json',metadata)
    digest=sha((ZEN/name).read_bytes())
    (ZEN/'SHA256SUMS.txt').write_text(digest+'  '+name+'\n',encoding='utf-8')
    manual=(BACKUP/'zenodo/Zenodo_填写信息.md').read_text(encoding='utf-8')
    manual=re.sub(r'SHA-256：`[a-f0-9]+`',f'SHA-256：`{digest}`',manual,count=1)
    manual=re.sub(r'（[\d,]+字节）',f'（{(ZEN/name).stat().st_size:,}字节）',manual,count=1)
    for heading,next_heading,value in [('Title','Description',metadata['title']),('Description','Version',metadata['description'])]:
        a=manual.index('## '+heading+'\n\n')+len('## '+heading+'\n\n');b=manual.index('\n\n## '+next_heading,a)
        manual=manual[:a]+value+manual[b:]
    manual+='\n## GitHub精确版本\n\n'+RELEASE+'\n\n公共下载验证以项目 docs/revise/PUBLIC_ACCESS_VERIFICATION.json 为准；Zenodo仍需上传到现有预留DOI草稿。\n'
    (ZEN/'Zenodo_填写信息.md').write_text(manual,encoding='utf-8')
    # A generic external filename avoids presenting internal project identifiers.
    release_dir=STAGE/'release';release_dir.mkdir(exist_ok=True)
    shutil.copy2(ZEN/name,release_dir/'DFSC_Revision_Reproducibility_20261025.zip')
    (release_dir/'SHA256SUMS.txt').write_text(digest+'  DFSC_Revision_Reproducibility_20261025.zip\n',encoding='utf-8')
    put(ZEN/'build_verification.json',dict(sha256=digest,bytes=(ZEN/name).stat().st_size,files=len(payload),
        current_paper_source_match=True,targeted_validation_integrated=True,zenodo_uploaded=False,zenodo_published=False,
        actual_archive_extracted_and_manifest_verified=False,github_release=RELEASE))
    print(json.dumps(dict(archive=name,sha256=digest,bytes=(ZEN/name).stat().st_size,files=len(payload))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','editorial','finalize','package'])
    args=parser.parse_args();globals()[args.phase]()
