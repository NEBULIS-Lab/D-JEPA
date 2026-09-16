"""Portable path and driving-log identities; no model imports."""
import re
from pathlib import PurePosixPath


def safe_relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe relative path: {value}')
    return str(path)


def log_group(name):
    return re.sub(r'_\d{5}_\d{5}$', '', name)
