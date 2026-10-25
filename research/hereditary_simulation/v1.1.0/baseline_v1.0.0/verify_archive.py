"""Verify the extracted public archive; no third-party dependencies needed."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Unsafe manifest path: {name}')
        data = path.read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError(f'Archive file mismatch: {name}')
    return {'all_checks_pass': True, 'verified_files': len(manifest['files']),
            'verification_scope': 'File hashes, not new scientific audit or global correctness.'}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
