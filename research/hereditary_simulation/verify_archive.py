"""Verify the complete public archive (standard library only)."""
from pathlib import Path
import hashlib,json
def verify(root):
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name,entry in m['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root):raise ValueError('Unsafe manifest path')
        data=p.read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('File mismatch: '+name)
    return {'all_checks_pass':True,'verified_files':len(m['files']),'scope':'File integrity only; not a new mathematical or material validation.'}
if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent),indent=2))
