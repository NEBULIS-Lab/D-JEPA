"""Download selected D-JEPA profiles / supervision with pinned revisions and checksums.

Install download dependencies: pip install -e '.[download]'
Files stay in the requested destination. Existing files are verified, never replaced.
The HF cache handles interrupted transfers; rerun the same command to resume.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil


def verify_sha256(path, expected):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f'checksum mismatch for {path}: {actual} != {expected}')
    return actual


def copy_verified(source, destination, expected):
    verify_sha256(source, expected)
    destination = Path(destination)
    if destination.exists():
        verify_sha256(destination, expected)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as target, Path(source).open('rb') as stream:
        shutil.copyfileobj(stream, target)
    verify_sha256(destination, expected)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--profile', action='append', default=[], help='Repeat to select multiple checkpoint profiles')
    parser.add_argument('--dataset', action='store_true', help='Download the single supervision ZIP (no automatic extraction)')
    parser.add_argument('--revision', default='main', help='Model repository revision; resolved to an immutable commit')
    parser.add_argument('--dataset-revision', default='main')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    if not args.profile and not args.dataset:
        parser.error('select --profile and/or --dataset')
    if any(not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', p) for p in args.profile):
        parser.error('invalid profile name')
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    if args.profile:
        repo = 'Shuaijun/D-JEPA'
        revision = api.model_info(repo, revision=args.revision).sha
        manifest = Path(hf_hub_download(repo, 'manifest.json', revision=revision))
        entries = {p['profile']: p for p in json.loads(manifest.read_text())['profiles']}
        for profile in args.profile:
            if profile not in entries:
                parser.error(f'profile not in release manifest: {profile}')
            for filename in ['config.json', 'README.md', 'model.pt']:
                source = Path(hf_hub_download(repo, f'{profile}/{filename}', revision=revision))
                expected = entries[profile]['weights_sha256'] if filename == 'model.pt' else hashlib.sha256(source.read_bytes()).hexdigest()
                if filename == 'config.json' and json.loads(source.read_text()) != entries[profile]:
                    raise ValueError(f'config differs from release manifest: {profile}')
                copy_verified(source, args.checkpoint_dir/profile/filename, expected)
            receipt = args.checkpoint_dir/profile/'download-receipt.json'
            if not receipt.exists():
                with receipt.open('x') as stream:
                    json.dump({'repo':repo,'revision':revision,'profile':profile,
                               'weights_sha256':entries[profile]['weights_sha256']}, stream, indent=2)
    if args.dataset:
        repo = 'Shuaijun/D-JEPA-Dataset'
        revision = api.dataset_info(repo, revision=args.dataset_revision).sha
        manifest_path = Path(hf_hub_download(repo, 'manifest.json', repo_type='dataset', revision=revision))
        manifest = json.loads(manifest_path.read_text())
        name = manifest['archive']
        if PurePosixPath(name).name != name or '\\' in name or name in ('.', '..'):
            raise ValueError('unsafe archive filename')
        source = Path(hf_hub_download(repo, name, repo_type='dataset', revision=revision))
        if source.stat().st_size != manifest['bytes']:
            raise ValueError('archive byte count mismatch')
        copy_verified(source, args.data_dir/name, manifest['sha256'])
        copy_verified(manifest_path, args.data_dir/'manifest.json', hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        receipt = args.data_dir/'download-receipt.json'
        if not receipt.exists():
            with receipt.open('x') as stream:
                json.dump({'repo':repo,'revision':revision, **manifest}, stream, indent=2)


if __name__ == '__main__':
    main()
