"""Check the portable CPU installation and explicitly requested local artifacts."""
import argparse
import importlib
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append', default=[])
    parser.add_argument('--split', type=Path, action='append', default=[])
    args = parser.parse_args()
    checks = []
    def check(name, action):
        try:
            detail = action()
            checks.append({'name': name, 'ok': True, 'detail': detail})
        except Exception as error:
            checks.append({'name': name, 'ok': False, 'error': str(error)})
    for name in ('numpy', 'torch', 'yaml', 'djepa'):
        check(name, lambda name=name: getattr(importlib.import_module(name), '__version__', 'importable'))
    def profile(path):
        from djepa.data.validation import sha256
        config = json.loads((path/'config.json').read_text())
        weights = path/'model.pt'
        if not weights.is_file():
            raise FileNotFoundError(weights)
        if config.get('weights_sha256') and sha256(weights) != config['weights_sha256']:
            raise ValueError('checkpoint SHA-256 mismatch')
        return {'architecture': config['architecture'], 'weights_present': True}
    for path in args.checkpoint:
        check(f'checkpoint:{path}', lambda path=path: profile(path))
    for path in args.split:
        def split(path=path):
            from djepa.data.validation import validate_split
            return validate_split(path)
        check(f'split:{path}', split)
    ok = all(c['ok'] for c in checks) and sys.version_info >= (3, 10)
    print(json.dumps({'ok': ok, 'python': platform.python_version(),
                      'scope': 'CPU package/artifact preflight; no GPU or simulator probe', 'checks': checks}, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
