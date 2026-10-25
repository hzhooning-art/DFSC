"""Strict finite records; atomic progress, exclusive immutable terminal results."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write_checkpoint(path, payload):
    path=Path(path)
    content=encode(payload)  # Validate before touching the old checkpoint.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name=tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_terminal(path, payload):
    path=Path(path)
    content=encode(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted exclusive terminal write is an error, never silently re-run.
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def verify_bindings(root):
    root=Path(root)
    receipt=read(root/'freeze_receipt.json')
    for name, binding in receipt['bindings'].items():
        path=Path(binding['path'])
        if not path.is_absolute():
            path=root/path
        if not path.is_file() or sha(path)!=binding['sha256']:
            raise ValueError('Frozen input changed: '+name)
    for name,version in receipt.get('versions',{}).items():
        if importlib.metadata.version(name)!=version:
            raise ValueError('Frozen dependency version changed: '+name)
    for module,binding in receipt.get('installed_source_hashes',{}).items():
        spec=importlib.util.find_spec(module)
        if spec is None or sha(spec.origin)!=binding['sha256']:
            raise ValueError('Frozen installed source changed: '+module)


def exception_record(exc):
    return {'error_type':type(exc).__name__, 'error':str(exc)}
