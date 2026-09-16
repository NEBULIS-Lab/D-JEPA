"""Bounded relative-benefit/risk alignment on fixed candidate caches.

Diagnostic outcomes are loaded after cross-log calibration is locked.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from .runtime import log_group
from .contract import write_json, write_csv

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEVICE = "cpu"


class RiskAligner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(dim,64),nn.LayerNorm(64),nn.GELU())
        layer=nn.TransformerEncoderLayer(64,4,128,dropout=0,activation='gelu',batch_first=True)
        self.body=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.gain_head=nn.Linear(64,1,bias=False)
        self.risk_head=nn.Linear(64,2)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.zeros_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    def forward(self,x,native):
        h=self.body(self.project(x))
        anchor=h[torch.arange(len(h),device=h.device),native.argmax(-1)]
        gain=.2*self.gain_head(h-anchor[:,None]).squeeze(-1).tanh()
        return gain,self.risk_head(h)


def log_folds(groups):
    if len(set(groups))<3:
        raise ValueError('Need at least three source logs for group calibration')
    return [([i for i,g in enumerate(groups) if g!=held],
             [i for i,g in enumerate(groups) if g==held]) for held in sorted(set(groups))]


def select(native,gain,risk,alpha,risk_weight,margin):
    anchor=native.argmax(-1)
    n=np.arange(len(native))
    # NC and TTC predictions are learned from actual fitting labels.
    burden=risk[...,0]+(5/12)*risk[...,1]
    penalty=np.maximum(burden-burden[n,anchor,None],0)
    score=native+alpha*gain-risk_weight*penalty
    chosen=score.argmax(-1)
    switch=score[n,chosen]>score[n,anchor]+margin+1e-8
    return np.where(switch,chosen,anchor)


def boundary_weights(native,y,risk):
    anchor=native.argmax(-1)
    ids=torch.arange(len(native),device=native.device)
    gap=y-y[ids,anchor,None]
    risk_increase=(risk-risk[ids,anchor,None]).clamp_min(0).amax(-1)
    near=torch.zeros_like(native).scatter_(1,native.topk(min(8,native.shape[1]),dim=-1).indices,1)
    return (.15+.85*near)*(1+4*(-gap).clamp_min(0)+8*risk_increase)


def load_examples(cache,role):
    from .learn import load_cache
    examples,identity=load_cache(cache,role)
    for e in examples:
        # Remove five unsupervised auxiliary logits; keep native, tie-aware rank
        # and actual proposal geometry. Never use future/scorer labels as inputs.
        e['features']=np.column_stack((e['x'][:,:256],e['x'][:,262:])).astype('float32')
        e['risk']=np.array([[1-r['no_at_fault_collisions'],1-r['time_to_collision_within_bound']]
                            for r in e['labels']],dtype='float32')
    return examples,identity


def arrays(examples,mean=None,std=None):
    x=np.stack([e['features'] for e in examples])
    if mean is None:
        mean=x.reshape(-1,x.shape[-1]).mean(0)
        std=np.maximum(x.reshape(-1,x.shape[-1]).std(0),1e-4)
    return (torch.as_tensor((x-mean)/std,device=DEVICE),
            torch.as_tensor(np.stack([e['native'] for e in examples]),device=DEVICE),
            torch.as_tensor(np.stack([e['y'] for e in examples]),device=DEVICE),
            torch.as_tensor(np.stack([e['risk'] for e in examples]),device=DEVICE),mean,std)


def fit(examples,seed,variant,epochs,snapshots,validation=None):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    x,native,y,risk,mean,std=arrays(examples)
    model=RiskAligner(x.shape[-1]).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    validation_x=None
    if validation is not None:
        validation_x=arrays(validation,mean,std)
    checkpoints,predictions,history={},{},[]
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for ix in torch.randperm(len(x),device=DEVICE).split(64):
            xx,ss,yy,rr=x[ix],native[ix],y[ix],risk[ix]
            gain,logits=model(xx,ss)
            anchor=ss.argmax(-1);bi=torch.arange(len(ss),device=DEVICE)
            delta=yy-yy[bi,anchor,None]
            predicted_delta=ss-ss[bi,anchor,None]+gain
            weights=boundary_weights(ss,yy,rr)
            regression=(weights*F.smooth_l1_loss(predicted_delta,delta.clamp(-.2,.2),beta=.05,reduction='none')).sum()/weights.sum()
            mask=delta.abs()>.005
            ranking=(weights*F.softplus(-delta.sign()*predicted_delta/.05)*mask).sum()/(weights*mask).sum().clamp_min(1)
            safety=(F.binary_cross_entropy_with_logits(logits,rr,reduction='none')*(1+3*rr)).mean()
            loss=regression+.1*ranking+.1*gain.square().mean()
            if variant=='risk_relative':loss=loss+.5*safety
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            losses.append(float(loss.detach()))
        history.append({'epoch':epoch,'loss':float(np.mean(losses))})
        if epoch in snapshots:
            model.eval()
            checkpoints[epoch]={'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                'mean':mean,'std':std,'dim':x.shape[-1]}
            if validation_x is not None:
                with torch.inference_mode():
                    g,r=model(validation_x[0],validation_x[1])
                predictions[epoch]=(g.cpu().numpy(),r.sigmoid().cpu().numpy())
    return checkpoints,predictions,history


def metrics(examples,chosen):
    anchor=np.array([e['native'].argmax() for e in examples])
    truth=np.stack([e['y'] for e in examples]);i=np.arange(len(truth))
    delta=truth[i,chosen]-truth[i,anchor]
    risk=np.stack([e['risk'] for e in examples])
    groups=np.array([log_group(e['row']['log_name']) for e in examples])
    group_delta=[float(delta[groups==g].mean()) for g in sorted(set(groups))]
    return {'N':len(examples),'PDMS':float(truth[i,chosen].mean()*100),
        'delta_PDMS':float(delta.mean()*100),'NC':float((1-risk[i,chosen,0]).mean()*100),
        'TTC':float((1-risk[i,chosen,1]).mean()*100),'gain':int((delta>1e-6).sum()),
        'loss':int((delta< -1e-6).sum()),'tie':int((abs(delta)<=1e-6).sum()),
        'switched':int((chosen!=anchor).sum()),'new_NC_failures':int((risk[i,chosen,0]>risk[i,anchor,0]+1e-6).sum()),
        'new_TTC_failures':int((risk[i,chosen,1]>risk[i,anchor,1]+1e-6).sum()),
        'mean_downside':float(np.minimum(delta,0).mean()),'worst_log_delta':min(group_delta)*100,
        'log_delta_std':float(np.std(group_delta))}


def calibrate(examples,predictions,variant):
    native=np.stack([e['native'] for e in examples]);anchor=native.argmax(-1)
    reference=metrics(examples,anchor)
    all_rows=[];best=None;best_value=-np.inf
    for epoch,(gain,risk) in sorted(predictions.items()):
        for alpha in (0.,.1,.25,.5,1.):
            for risk_weight in ((0.,.1,.3) if variant=='risk_relative' else (0.,)):
                for margin in (0.,.005,.02):
                    chosen=select(native,gain,risk,alpha,risk_weight,margin)
                    result=metrics(examples,chosen)
                    # Constraints use ONLY out-of-fold examples. Keep native fallback.
                    admissible=result['NC']>=reference['NC']-1e-6 and result['TTC']>=reference['TTC']-1e-6
                    utility=result['delta_PDMS']/100+.25*result['mean_downside']-.10*result['log_delta_std']
                    setting={'epoch':epoch,'alpha':alpha,'risk_weight':risk_weight,'margin':margin}
                    all_rows.append({**setting,**result,'admissible':admissible,'utility':utility})
                    if admissible and utility>best_value+1e-10:
                        best_value=utility;best={**setting,'OOF_metrics':result,'utility':utility}
    assert best is not None
    return best,all_rows


def main():
    global DEVICE
    p=argparse.ArgumentParser()
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a=p.parse_args()
    DEVICE=a.device
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.time()
    snapshots=(1,) if a.smoke else (8,16,32)
    config={'seed':a.seed,'matrix':str(a.matrix.resolve()),'variants':['boundary_only','risk_relative'],
        'snapshots':snapshots,'lr':3e-4,'batch_size':64,'weight_decay':1e-3,
        'feature_schema':'query256_native_rank_trajectory24_no_unverified_subscores',
        'protocol':'leave_one_log_out_calibration; held_out_diagnostic_replay',
        'smoke':a.smoke,'test_used_for_training_or_calibration':False}
    write_json(out/'config.json',config)
    source_files=['risk.py','learn.py','alignment.py','contract.py']
    write_json(out/'meta.json',{'command':sys.argv,'device':DEVICE,
        'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
        'sources':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_files}})
    from .learn import assert_identity
    fitting,identity=load_examples(a.matrix/'fit','fit')
    calibration,cal_identity=load_examples(a.matrix/'calibration','calibration')
    assert_identity(identity,cal_identity)
    examples=fitting+calibration
    groups=[log_group(e['row']['log_name']) for e in examples]
    folds=log_folds(groups)
    write_json(out/'folds.json',[{'train_tokens':[examples[i]['row']['token'] for i in tr],
        'validation_tokens':[examples[i]['row']['token'] for i in va],
        'validation_log':groups[va[0]]} for tr,va in folds])
    selected={};trained={};training_history=[]
    for variant in config['variants']:
        oof={ep:(np.zeros((len(examples),32),dtype='float32'),np.zeros((len(examples),32,2),dtype='float32')) for ep in snapshots}
        for fold,(tr,va) in enumerate(folds):
            checkpoints,preds,history=fit([examples[i] for i in tr],a.seed+fold*100,variant,max(snapshots),snapshots,[examples[i] for i in va])
            for ep in snapshots:
                oof[ep][0][va]=preds[ep][0];oof[ep][1][va]=preds[ep][1]
            training_history.extend({'variant':variant,'fold':fold,**h} for h in history)
            print(f'{variant} fold {fold+1}/{len(folds)} complete',flush=True)
        chosen,table=calibrate(examples,oof,variant)
        write_json(out/f'{variant}_selection.json',chosen)
        write_csv(out/f'{variant}_calibration_grid.csv',table)
        np.savez_compressed(out/f'{variant}_oof.npz',**{f'{kind}_{ep}':v for ep,pair in oof.items() for kind,v in zip(('gain','risk'),pair)})
        final,_,history=fit(examples,a.seed,variant,chosen['epoch'],[chosen['epoch']])
        trained[variant]=final[chosen['epoch']]
        selected[variant]=chosen
        torch.save({'model':trained[variant],'selection':chosen,'config':config,
                    'fit_source_identity':identity,'fit_count':len(examples),'test_used':False},out/f'{variant}.pt')
        training_history.extend({'variant':variant,'fold':'refit',**h} for h in history)
        print('CALIBRATION_LOCKED',variant,json.dumps(chosen),flush=True)
    write_json(out/'selections_locked_before_test.json',selected)
    # Only now load previously inspected test outcomes for diagnostic replay.
    diagnostic,d_identity=load_examples(a.matrix/'test','test')
    assert_identity(identity,d_identity)
    summary={'state':'complete','seed':a.seed,'protocol':config['protocol'],'new_independent_validation':False,
             'methods':{},'duration_seconds':None,'source_matrix':str(a.matrix.resolve())}
    native=np.stack([e['native'] for e in diagnostic]);native_idx=native.argmax(-1)
    summary['methods']['native']=metrics(diagnostic,native_idx)
    raw=[]
    for variant in config['variants']:
        st=trained[variant];model=RiskAligner(st['dim']).to(DEVICE).eval();model.load_state_dict(st['state'])
        x,s,_,_,_,_=arrays(diagnostic,st['mean'],st['std'])
        with torch.inference_mode():g,r=model(x,s)
        setting=selected[variant]
        chosen=select(native,g.cpu().numpy(),r.sigmoid().cpu().numpy(),setting['alpha'],setting['risk_weight'],setting['margin'])
        summary['methods'][variant]={**metrics(diagnostic,chosen),'calibration':setting}
        for i,ex in enumerate(diagnostic):
            raw.append({'token':ex['row']['token'],'log_name':ex['row']['log_name'],'variant':variant,
                'candidate_id':int(chosen[i]),'baseline_id':int(native_idx[i]),
                'PDMS':float(ex['y'][chosen[i]]*100),'baseline_PDMS':float(ex['y'][native_idx[i]]*100),
                'NC':float(1-ex['risk'][chosen[i],0]),'TTC':float(1-ex['risk'][chosen[i],1]),
                'identity':'original_test_now_diagnostic'})
    write_csv(out/'training_history.csv',training_history)
    write_csv(out/'diagnostic_results.csv',raw)
    summary['duration_seconds']=time.time()-start
    write_json(out/'summary.json',summary)
    print('RUN_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
