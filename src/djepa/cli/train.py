"""Portable ordinal-instance training on released train/calibration features."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from ..models.ordinal import DecisionAligner
from ..objectives.ordinal import decision_loss
from ..evaluation.inference import gated_positions
from ..data.supervision import require_disjoint, load_split
from .config import parse_config, save_resolved_config


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


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=120)
    parser.add_argument('--seed',type=int,default=20260906)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--learning-rate',type=float,default=.0003)
    parser.add_argument('--weight-decay',type=float,default=.0001)
    parser.add_argument('--max-grad-norm',type=float,default=1.)
    parser.add_argument('--evaluate-every',type=int,default=10)
    args=parse_config(parser, argv)
    if min(args.epochs,args.batch_size,args.evaluate_every)<1:
        parser.error('epochs, batch-size and evaluate-every must be positive')
    if not np.isfinite([args.learning_rate,args.weight_decay,args.max_grad_norm]).all() or args.learning_rate<=0 or args.weight_decay<0 or args.max_grad_norm<=0:
        parser.error('invalid optimizer hyperparameters')
    if args.output.exists():
        raise FileExistsError(f'output already exists: {args.output}')
    train,cal=load_split(args.train),load_split(args.calibration)
    require_disjoint(train.get('base_ids',train['start_ids']),cal.get('base_ids',cal['start_ids']))
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=DecisionAligner(3)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    best=calibrate(model,cal);state=copy.deepcopy(model.state_dict());best_epoch=0
    for epoch in range(1,args.epochs+1):
        model.train()
        for indices in torch.randperm(len(train['features'])).split(args.batch_size):
            indices=indices.numpy()
            scores,correction=model(torch.tensor(train['features'][indices],dtype=torch.float32),
                                    torch.tensor(train['base_scores'][indices],dtype=torch.float32))
            loss=decision_loss(scores,correction,torch.tensor(train['success'][indices]))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm);optimizer.step()
        if epoch%args.evaluate_every==0 or epoch==args.epochs:
            candidate=calibrate(model,cal)
            if candidate[:2]>best[:2]:best,state,best_epoch=candidate,copy.deepcopy(model.state_dict()),epoch
            print(json.dumps({'epoch':epoch,'calibration_success':candidate[0]}),flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    torch.save(state,args.output/'model.pt')
    config={'project':'D-JEPA','architecture':'set_aligner','input_dim':3,'gate_mode':'refined_gap',
            'gate_threshold':best[2],'max_correction':.2,'seed':args.seed,'selected_epoch':best_epoch,
            'training_mode':'portable CPU trainer; exact historical checkpoints supplied separately'}
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    save_resolved_config(args, args.output/'resolved_config.json')


if __name__=='__main__':main()
