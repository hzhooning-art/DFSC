"""Read source XLSX only; write provenance-preserving JSON, never alter Excel."""
from pathlib import Path
import io,json,zipfile,hashlib,math
import openpyxl
ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'experiments/specs/orbital_hold_pilot_20261021.json'
OUT=ROOT/'results/orbital_hold_pilot_20261021'
def main():
    spec=json.loads(SPEC.read_text());OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'extracted.json'
    if dest.exists():raise FileExistsError('Do not overwrite extracted source')
    archive=next((ROOT/'data/external/orbital_fat_20261021').glob('*.zip'))
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==spec['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        name=next(n for n in z.namelist() if not n.startswith('__MACOSX') and n.endswith('Stress relaxation_V02.xlsx'))
        raw=z.read(name)
    book=openpyxl.load_workbook(io.BytesIO(raw),read_only=True,data_only=True)
    traces=[]
    for name in spec['individual_sheets']:
        sheet=book[name];rows=list(sheet.values)
        assert rows[0][:3]==('time','modulus G(t)','shear stress') and rows[1][:3]==('s','Pa','Pa')
        values=[];excluded=[]
        for index,row in enumerate(rows[3:],4):
            t,g,stress=row[:3]
            if all(isinstance(x,(float,int)) and math.isfinite(x) for x in (t,g,stress)) and t>=0:
                values.append({'source_row':index,'time_seconds':t,'modulus_Pa':g,'stress_Pa':stress})
            else:excluded.append(index)
        traces.append({'sheet':name,'raw_records':values,'non_numeric_rows':excluded})
    receipt={'spec_sha256':hashlib.sha256(SPEC.read_bytes()).hexdigest(),'archive_sha256':spec['archive_sha256'],
        'xlsx_member':name,'xlsx_sha256':hashlib.sha256(raw).hexdigest(),'traces':traces,
        'individual_sheet_count':len(traces),'independent_donor_count':None,
        'excluded_summary_sheets':[n for n in book.sheetnames if n not in spec['individual_sheets']]}
    # Preserve the archive member path, not the last sheet name.
    receipt['xlsx_member']='Edited Human Fat Shear Viscoelasticity Data for Deposit/Stress relaxation_V02.xlsx'
    dest.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'traces':len(traces),'records':[len(t['raw_records']) for t in traces]}))
if __name__=='__main__':main()
