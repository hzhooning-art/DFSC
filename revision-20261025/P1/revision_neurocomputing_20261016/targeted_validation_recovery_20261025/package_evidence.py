"""Standalone source/data archive, safe extraction, exact-extraction replay tests."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import stat
import subprocess
import sys
import tempfile
import zipfile

from freeze_protocol import HERE,REV
from records import sha,read,write_terminal,verify_bindings


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(set(names))!=len(names):
            raise ValueError('Duplicate ZIP paths')
        for info in archive.infolist():
            p=PurePosixPath(info.filename.replace('\\','/'))
            if p.is_absolute() or '..' in p.parts or ':' in info.filename or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('Unsafe archive member '+info.filename)
        manifest=json.loads(archive.read('EVIDENCE_MANIFEST.json'))
        if set(names)!=set(manifest)|{'EVIDENCE_MANIFEST.json'}:
            raise ValueError('ZIP/manifest membership mismatch')
        for name,digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:
                raise ValueError('Archive digest mismatch: '+name)
        bad=archive.testzip()
        if bad:
            raise ValueError('Corrupt ZIP member '+bad)
    return manifest


def build():
    verify_bindings(HERE)
    if not (HERE/'results/audit.json').is_file():
        raise ValueError('Audit required before packaging')
    repo=HERE.parents[2]
    files={}
    for path in HERE.rglob('*'):
        rel=path.relative_to(HERE)
        if path.is_file() and not any(part in ('tmp','__pycache__','output') for part in rel.parts) and path.name!='execution.lock':
            files[path.relative_to(repo).as_posix()]=path
    for binding in read(HERE/'freeze_receipt.json')['bindings'].values():
        path=(HERE/binding['path']).resolve()
        files[path.relative_to(repo).as_posix()]=path
    # Preserve the original experiment and dependencies as provenance, including
    # failed/partial outcomes. Do not silently replace it with only the new run.
    prior=read(HERE/'prior_preservation.json')
    old_root=HERE.parent/'targeted_validation_20261023'
    for rel,digest in prior['files'].items():
        path=old_root/rel
        if sha(path)!=digest:
            raise ValueError('Original evidence changed: '+rel)
        files[path.relative_to(repo).as_posix()]=path
    for binding in read(old_root/'freeze_receipt.json')['bindings'].values():
        path=(old_root/binding['path']).resolve()
        if sha(path)!=binding['sha256']:
            raise ValueError('Original frozen dependency changed')
        files[path.relative_to(repo).as_posix()]=path
    license_path=REV/'cpu_repair_delivery_20261021/backend/LICENSE'
    files[license_path.relative_to(repo).as_posix()]=license_path
    manifest={name:sha(path) for name,path in sorted(files.items())}
    out=HERE/'output/DFSC_Targeted_Validation_Recovery_20261025.zip'
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,path in sorted(files.items()):
            archive.write(path,name)
        archive.writestr('EVIDENCE_MANIFEST.json',json.dumps(manifest,indent=2))
    verified=verify_archive(out)
    with tempfile.TemporaryDirectory(prefix='targeted-evidence-') as tmp:
        extract=Path(tmp)
        with zipfile.ZipFile(out) as archive:
            archive.extractall(extract)  # Only after every path/hash has been checked.
        for name,digest in verified.items():
            if sha(extract/name)!=digest:
                raise ValueError('Extracted hash mismatch')
        test_root=extract/HERE.relative_to(repo)/'tests'
        command=[sys.executable,'-B','-X','utf8','-m','unittest','discover','-s',str(test_root),'-p','test_*.py']
        tests=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',cwd=extract,timeout=180)
        receipt={'archive':out.name,'sha256':sha(out),'bytes':out.stat().st_size,
                 'manifest_members':len(verified),'extracted_member_hashes_verified':True,
                 'replay_command':command,'replay_exit_code':tests.returncode,
                 'replay_stdout':tests.stdout,'replay_stderr':tests.stderr,
                 'fresh_dependency_install_claimed':False,'public_or_portal_write':False}
        write_terminal(HERE/'output/packaging_verification.json',receipt)
        if tests.returncode:
            raise RuntimeError('Exact-extraction replay failed; see receipt')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('replay_stdout','replay_stderr','replay_command')}))
    return receipt


if __name__=='__main__':
    build()
