"""Strict YAML defaults with explicit command-line overrides."""
import argparse
import json
from pathlib import Path

import yaml


class UniqueLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError('configuration keys must be unique strings')
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def parse_config(parser, argv=None):
    """Paths are relative to the working directory, equally for YAML and CLI."""
    parser.add_argument('--config', type=Path, help='YAML defaults; CLI options take precedence')
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config', type=Path)
    known, _ = probe.parse_known_args(argv)
    if known.config:
        try:
            data = yaml.load(known.config.read_text(), Loader=UniqueLoader)
            if not isinstance(data, dict):
                raise ValueError('expected a mapping')
            actions = {a.dest: a for a in parser._actions if a.dest not in ('help', 'config')}
            for key, value in data.items():
                if key not in actions:
                    raise ValueError(f'unknown field: {key}')
                action = actions[key]
                if action.type is int and type(value) is not int:
                    raise ValueError(f'{key} must be an integer')
                if action.type is float and (type(value) not in (int, float)):
                    raise ValueError(f'{key} must be numeric')
                if action.type is Path and (not isinstance(value, str) or not value):
                    raise ValueError(f'{key} must be a nonempty path')
                data[key] = action.type(value) if action.type else value
                action.required = False
            parser.set_defaults(**data)
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            parser.error(f'invalid configuration: {exc}')
    return parser.parse_args(argv)


def save_resolved_config(args, path):
    with Path(path).open('x') as stream:
        json.dump(vars(args), stream, default=str, indent=2)
        stream.write('\n')
