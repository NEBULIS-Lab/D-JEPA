"""Validate supervision schemas; optionally hash the extracted release manifest."""
import argparse
import json
from pathlib import Path

from djepa.data.validation import validate_manifest, validate_split


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/D-JEPA-supervision-v1'))
    parser.add_argument('--split', type=Path, help='One split instead of discovering all inputs.npz')
    parser.add_argument('--checkpoint', type=Path, help='Check feature dimensions for --split')
    parser.add_argument('--checksums', action='store_true', help='Read every file named in the release manifest')
    args = parser.parse_args()
    if args.checkpoint and not args.split:
        parser.error('--checkpoint requires --split')
    paths = [args.split] if args.split else sorted(p.parent for p in args.data_root.glob('*/*/inputs.npz'))
    if not paths:
        parser.error('no dataset splits found')
    report = {'splits': {str(p): validate_split(p, args.checkpoint) for p in paths}}
    if args.checksums:
        report['manifest'] = validate_manifest(args.data_root)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
