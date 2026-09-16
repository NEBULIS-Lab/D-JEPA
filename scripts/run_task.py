#!/usr/bin/env python3
"""Run a task recipe; command-line arguments override YAML defaults."""
import argparse
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULES = {'djepa.driving.learn', 'djepa.driving.risk', 'djepa.driving.native', 'djepa.driving.score'}


def command(config, extra):
    if set(config) != {'entrypoint', 'arguments'}:
        raise ValueError('Recipe requires entrypoint and arguments')
    entry = config['entrypoint']
    if entry in MODULES:
        result = [sys.executable, '-m', entry]
    else:
        path = (ROOT / entry).resolve()
        if not path.is_relative_to(ROOT / 'scripts/robotics') or not path.is_file():
            raise ValueError('Unsupported task entrypoint')
        result = [sys.executable, str(path)]
    for key, value in config['arguments'].items():
        if not isinstance(key, str) or key.startswith('-'):
            raise ValueError('Argument names omit leading dashes')
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif value is not None:
            result += [flag] + [str(v) for v in (value if isinstance(value, list) else [value])]
    return result + extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    with args.config.open() as handle:
        cmd = command(yaml.safe_load(handle), extra)
    if args.dry_run:
        import shlex
        print(shlex.join(cmd))
        return 0
    return subprocess.run(cmd, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
