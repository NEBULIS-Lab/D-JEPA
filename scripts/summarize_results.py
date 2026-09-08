"""Summarize cached evaluator outputs without pooling distinct populations."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Output prefix for .csv and .md')
    args = parser.parse_args()
    fields = ['report', 'mode', 'count', 'observed_count', 'unobserved_count',
              'success_count', 'success_rate_on_observed', 'full_population_success_rate']
    rows = []
    for path in args.reports:
        data = json.loads(path.read_text())
        if not data.get('mode', '').startswith('cached-feature checkpoint replay'):
            parser.error(f'{path}: expected an evaluator output, not a raw historical authority')
        row = {k: data.get(k) for k in fields}
        row['report'] = str(path)
        row['count'] = data.get('count', len(data.get('selected_ids', [])))
        rows.append(row)
    paths = [args.output.with_suffix('.csv'), args.output.with_suffix('.md')]
    if any(p.exists() for p in paths):
        raise FileExistsError('summary output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with paths[1].open('x') as stream:
        stream.write('# Cached-decision replay summaries\n\nEach row is a separate population; no pooled average. Blank metrics mean unavailable.\n\n')
        stream.write('| ' + ' | '.join(fields) + ' |\n|' + '|'.join(['---']*len(fields)) + '|\n')
        for row in rows:
            stream.write('| ' + ' | '.join('' if row[k] is None else str(row[k]).replace('|', '\\|').replace('\n', ' ') for k in fields) + ' |\n')


if __name__ == '__main__':
    main()
