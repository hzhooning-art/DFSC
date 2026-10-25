"""Independent XLSX XML extraction and high-precision positive-spectrum audit."""
from pathlib import Path
import hashlib,json,zipfile,xml.etree.ElementTree as ET
import numpy as np
import mpmath as mp
from prediction_time_search import decision
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/polymer_glass_hold_20261021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    mp.mp.dps=80;m=lambda x:mp.mpf(float(x))
    spec=json.loads((ROOT/'experiments/specs/polymer_glass_hold_20261021.json').read_text())
    result=json.loads((OUT/'result.json').read_text());receipt=json.loads((OUT/'protocol_receipt.json').read_text())
    extracted=json.loads((OUT/'extracted.json').read_text());arrays=np.load(OUT/'lp_witnesses.npz')
    assert all(sha(ROOT/k)==v for k,v in receipt['bindings'].items())
    assert sha(OUT/'lp_witnesses.npz')==result['lp_witnesses_sha256']
    workbook=ROOT/'data/external/polymer_glass_20261021'/spec['workbook']
    assert sha(workbook)==spec['workbook_sha256']
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(workbook) as z:
        book=ET.fromstring(z.read('xl/workbook.xml'));sheet=next(x for x in book.findall('m:sheets/m:sheet',ns) if x.attrib['name']==spec['sheet'])
        rid=sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'));target=next(x.attrib['Target'] for x in rels if x.attrib['Id']==rid)
        path=target.lstrip('/') if target.startswith('/') else 'xl/'+target
        xml=ET.fromstring(z.read(path));cells={c.attrib['r']:c for c in xml.findall('.//m:sheetData/m:row/m:c',ns)}
        source_pairs=0
        for trace in extracted['traces']:
            xc,yc=trace['source_columns'];assert xc<=9 and yc<=10
            for r in trace['records']:
                for col,key in ((xc,'time_seconds'),(yc,'modulus_MPa')):
                    coordinate=chr(64+col)+str(r['source_row']);cell=cells[coordinate]
                    assert cell.find('m:f',ns) is None and cell.attrib.get('t') not in ('s','str','inlineStr')
                    assert float(cell.find('m:v',ns).text)==r[key]
                source_pairs+=1
    assert source_pairs==251 and len(result['results'])==90
    lps=physical=cells_checked=tails=0;maxres=0.;accepted=[]
    for row in result['results']:
        tr=next(t for t in extracted['traces'] if t['temperature_K']==row['temperature_K'])
        t=np.array([r['time_seconds'] for r in tr['records']]);y=np.array([r['modulus_MPa'] for r in tr['records']])
        span=t[-1]-t[0];pick=lambda q:int(np.argmin(abs(t-q)))
        obs=[pick(t[0]+row['window_fraction']*f*span) for f in spec['observation_within_window_fractions']]
        idx=pick(t[0]+spec['target_span_fractions'][row['target_index']]*span)
        assert max(obs)<idx and np.array_equal(y[obs]/y[0],row['centers'])
        assert np.array_equal((t[obs]-t[0])/span,row['delays'])
        assert t[idx]==row['actual_seconds'] and y[idx]/y[0]==row['measured_response']
        assert y[idx]==row['measured_modulus_MPa'] and y[0]*row['band']==row['absolute_band_MPa']
        b=row['bracket']
        if b is None:assert row['decision']['state']=='unresolved';continue
        centers=np.array(row['centers']);delays=np.array(row['delays']);band=row['band'];rhs=np.r_[centers+band,-centers+band]
        for side,record in b['physical_witnesses'].items():
            stem=record['label'];nodes,x,dual,z=[arrays[stem+'_'+k] for k in ('nodes','x','y','z')]
            E=np.exp(-delays[:,None]*nodes);A=np.vstack((E,-E));c=np.exp(-row['wait']*nodes)*(1 if side=='il' else -1)
            residual=max(0.,float(np.max(A@x-rhs)),float(-np.min(x)),float(np.max(abs(c-A.T@dual-z))),
                float(np.max(dual)),float(-np.min(z)),float(abs(c@x-rhs@dual)))
            assert residual<=2e-7;maxres=max(maxres,residual);lps+=1
            pn,pw=arrays[stem+'_physical_nodes'],arrays[stem+'_physical_weights']
            for d,center in zip(delays,centers):
                response=mp.fsum(m(w)*mp.exp(-m(d)*m(rate)) for rate,w in zip(pn,pw))
                assert abs(response-m(center))<=m(band)+mp.mpf('1e-25')
            physical+=1
        for cert in b['certificates'].values():
            rates,coefs=list(map(m,cert['rates'])),list(map(m,cert['coefficients']))
            ev=lambda x:[mp.exp(-d*x) for d in rates]
            value=lambda x:mp.fsum(c*e for c,e in zip(coefs,ev(x)))
            previous=mp.mpf(0)
            for left,right,stored in cert['leaves']:
                l,r=m(left),m(right);assert l==previous and r>l;previous=r
                el,er=ev(l),ev(r)
                second=mp.fsum(c*d*d*(e1 if c>=0 else e0) for c,d,e0,e1 in zip(coefs,rates,el,er))
                bound=min(max(value(l),value(r))+max(mp.mpf(0),-second)*(r-l)**2/8,
                    mp.fsum(c*(e0 if c>=0 else e1) for c,e0,e1 in zip(coefs,el,er)))
                assert m(stored)>=bound-mp.mpf('1e-25') and cert['upper']>=stored;cells_checked+=1
            assert previous==m(cert['tail_start'])
            tail=mp.fsum(c for c,d in zip(coefs,rates) if not d)+mp.fsum(c*mp.exp(-d*previous) for c,d in zip(coefs,rates) if d>0 and c>0)
            assert m(cert['tail_upper'])>=tail-mp.mpf('1e-25') and cert['upper']>=cert['tail_upper'];tails+=1
        answer=decision(b,.05,.10,0.);assert answer==row['decision']
        assert row['measured_target_in_outer']==(b['outer'][0]<=row['measured_response']<=b['outer'][1])
        if answer['state']=='sufficient':
            q=answer['joint_prediction'];actual=row['measured_response']
            assert abs(q-actual)<=.05 and abs(q-actual)/abs(actual)<=.10
            accepted.append({'temperature_K':row['temperature_K'],'window_fraction':row['window_fraction'],
                'band_fraction':band,'target_seconds':row['actual_seconds'],'relative_error':abs(q-actual)/abs(actual)})
        for metric in row['point_baselines'].values():
            ae=abs(metric['prediction_scaled']-row['measured_response']);re=ae/abs(row['measured_response'])
            assert abs(ae-metric['absolute_error_scaled'])<1e-14 and abs(re-metric['relative_error'])<1e-14
            assert metric['joint_tolerance_met']==(ae<=.05 and re<=.10)
    assert lps==physical==len(result['lp_logs'])==180 and tails==180 and len(accepted)==1
    assert all(r['measured_target_in_outer'] for r in result['results'])
    audit={'all_numerical_checks_passed':True,'source_xml_replay_passed':True,'unmodified_measured_source_pairs':source_pairs,
        'source_fitted_columns_used':False,'lp_replays':lps,'physical_witnesses_at_80_digits':physical,
        'finite_cells_at_80_digits':cells_checked,'infinite_tails_at_80_digits':tails,'max_lp_residual':maxres,
        'result_sha256':sha(OUT/'result.json'),'accepted_exploratory_cells':accepted,
        'unique_physical_target_values':15,'window_specific_tasks_before_repeated_band_sensitivity':30,
        'reported_correlated_sensitivity_requests':90,'independent_replicate_specimens_available':False,
        'assumed_noise_bands_independently_calibrated':False,'calibrated_real_material_validation':False,
        'conditional_feasibility_positive_example':True,
        'scope':'One successful assumed-band cell and complete retained grid. This is not a success rate, a calibrated deterministic material guarantee, or superiority over classical continuous LP/point fits.'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
