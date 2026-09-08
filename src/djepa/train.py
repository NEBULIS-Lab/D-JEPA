"""Portable ordinal-instance training on released train/calibration features."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .ordinal import DecisionAligner, decision_loss
from .inference import gated_positions


def require_disjoint(train_ids, calibration_ids):
    left, right = set(np.asarray(train_ids).tolist()), set(np.asarray(calibration_ids).tolist())
    if not left or not right or left & right:
        raise ValueError('training and calibration base identities must be nonempty and disjoint')


def load_split(directory):
    with np.load(directory/'inputs.npz',allow_pickle=False) as a, np.load(directory/'labels.npz',allow_pickle=False) as b:
        data={k:a[k] for k in a.files}
        if 'supervision_mask' in b and not b['supervision_mask'].all():
            raise ValueError('ordinal trainer requires dense supervision; use granular masked objective for sparse data')
        data['success']=b['success']
        data['distance']=b['state_distance']
    if data['features'].shape[-1]!=3:
        raise ValueError('this trainer is the task-local three-coordinate ordinal instance')
    return data


@torch.no_grad()
def calibrate(model,data):
    model.eval()
    scores=np.concatenate([model(torch.tensor(data['features'][i:i+16],dtype=torch.float32),
                                  torch.tensor(data['base_scores'][i:i+16],dtype=torch.float32))[0].numpy()
                           for i in range(0,len(data['features']),16)])
    rows=np.arange(len(scores))
    candidates=[]
    for threshold in [0.,.005,.01,.02,.05,.1,.2,1.]:
        positions=gated_positions(data['base_scores'],scores,data['candidate_ids'],threshold,'refined_gap')
        candidates.append((int(data['success'][rows,positions].sum()),
                           -float(data['distance'][rows,positions].mean()),threshold))
    return max(candidates)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=120)
    parser.add_argument('--seed',type=int,default=20260906)
    args=parser.parse_args()
    if args.epochs<1: raise ValueError('epochs must be positive')
    train,cal=load_split(args.train),load_split(args.calibration)
    require_disjoint(train.get('base_ids',train['start_ids']),cal.get('base_ids',cal['start_ids']))
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=DecisionAligner(3)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    best=calibrate(model,cal);state=copy.deepcopy(model.state_dict());best_epoch=0
    for epoch in range(1,args.epochs+1):
        model.train()
        for indices in torch.randperm(len(train['features'])).split(16):
            indices=indices.numpy()
            scores,correction=model(torch.tensor(train['features'][indices],dtype=torch.float32),
                                    torch.tensor(train['base_scores'][indices],dtype=torch.float32))
            loss=decision_loss(scores,correction,torch.tensor(train['success'][indices]))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if epoch%10==0 or epoch==args.epochs:
            candidate=calibrate(model,cal)
            if candidate[:2]>best[:2]:best,state,best_epoch=candidate,copy.deepcopy(model.state_dict()),epoch
            print(json.dumps({'epoch':epoch,'calibration_success':candidate[0]}),flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    torch.save(state,args.output/'model.pt')
    config={'project':'D-JEPA','architecture':'set_aligner','input_dim':3,'gate_mode':'refined_gap',
            'gate_threshold':best[2],'max_correction':.2,'seed':args.seed,'selected_epoch':best_epoch,
            'training_mode':'portable CPU trainer; exact historical checkpoints supplied separately'}
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')


if __name__=='__main__':main()
