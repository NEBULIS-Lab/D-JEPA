"""Replay a frozen decision checkpoint; scoring never consumes outcome labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .inference import predict


def summarize_selections(positions, success, supervision_mask=None):
    positions, success = np.asarray(positions), np.asarray(success)
    if success.dtype.kind != 'b' or success.ndim != 2 or positions.shape != (len(success),):
        raise ValueError('Boolean outcome matrix and aligned selections required')
    mask = np.ones_like(success) if supervision_mask is None else np.asarray(supervision_mask)
    if mask.shape != success.shape or mask.dtype.kind != 'b':
        raise ValueError('Boolean supervision mask must align')
    rows = np.arange(len(success))
    observed = mask[rows, positions]
    count = int(observed.sum())
    total = int(success[rows, positions][observed].sum())
    return {'count':len(success), 'observed_count':count, 'unobserved_count':int((~observed).sum()),
            'success_count':total, 'success_rate_on_observed':total/count if count else None,
            'full_population_success_rate':total/count if count == len(success) else None}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--labels',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(2)
    with np.load(args.inputs,allow_pickle=False) as inputs:
        result=predict(args.checkpoint,inputs,args.batch_size)
    report={'selected_ids':result['selected_ids'].tolist(), 'mode':'cached-feature checkpoint replay; no new simulator execution'}
    if args.labels:
        with np.load(args.labels,allow_pickle=False) as labels:
            report.update(summarize_selections(result['selected_positions'],labels['success'],
                                               labels['supervision_mask'] if 'supervision_mask' in labels else None))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='selected_ids'}))


if __name__=='__main__':main()
