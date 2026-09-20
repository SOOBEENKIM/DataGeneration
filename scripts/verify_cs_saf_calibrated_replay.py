"""Independently sum saved transition arrays, never call production score/decomposition."""
import json
import numpy as np
import torch
from experiments.cs_saf_calibrated_replay import (ROOT,OUTPUT,CONFIG_SHA,contract,folder_for,verify,sha256,write_json)
from experiments import cs_saf_generation_repeats as parent


def main():
    c=contract();terminal=json.loads((OUTPUT/'GRID_COMPLETE.json').read_text())
    expected={(p,k,t) for p in c['prevalences'] for k in c['kappas'] for t in c['trials']}
    if {tuple(v) for v in terminal['completed']}!=expected:raise ValueError('incomplete grid')
    torch.set_num_threads(1);maxerr=0.;n=0;decompositions=0
    def close(a,b):
        nonlocal maxerr
        err=float(np.max(abs(np.asarray(a)-np.asarray(b))));maxerr=max(maxerr,err)
        if err>1e-12:raise ValueError(f'independent value mismatch {err}')
    for pi,k,t in sorted(expected):
        folder=folder_for(pi,k,t);m=verify(folder)
        if m['source_commit']!=terminal['source_commit']:raise ValueError('mixed sources')
        out=json.loads((folder/'replay.json').read_text());curves={};empirical={}
        for name in c['models']:
            for repeat in c['repeats']:
                key=f'{name}_repeat_{repeat}';p=parent.folder_for(pi,k,t)/name/f'repeat_{repeat}'
                path=torch.load(p/'generated_sample.pt',map_location='cpu')['sample']
                arrays=np.load(folder/(key+'.npz'));record=out['paths'][key]
                if sha256(p/'generated_sample.pt')!=record['identity']['sample_sha256']:raise ValueError('changed path')
                old=json.loads((p/'comparison.json').read_text())
                for group in c['groups']:
                    block=record['scores'][group];mask=path['valid_mask'].numpy().copy();mask[:,0]=False
                    if group!='pooled':mask&=(path['static_codes'].numpy()==3+int(group[-1]))[:,None]
                    edges=np.array(record['edges'][group]);gap=path['gap'].numpy().astype(float)
                    labels=np.sum(gap[:,:,None]>=edges,axis=2)
                    marks=path['receiver'].numpy();obs=(marks==np.roll(marks,1,axis=1)).astype(float)
                    means=[];counts=[];pred=[];tv=[]
                    for b in range(len(edges)+1):
                        ix=mask&(labels==b);counts.append(int(ix.sum()))
                        means.append(float(obs[ix].mean()) if ix.any() else 0.)
                        pred.append([float(arrays['repeat'][j][ix].astype(float).mean()) if ix.any() else 0. for j in (0,1)])
                        tv.append(float(arrays['mark_tv'][ix].mean()) if ix.any() else 0.)
                    pred=np.array(pred).T;ref=np.array(block['reference_counts']);counts_ref=ref.sum(1)
                    target=np.divide(ref[:,1],counts_ref,out=np.zeros(len(ref)),where=counts_ref>0);w=counts_ref/counts_ref.sum()
                    l=float(sum(w*abs(np.array(means)-target)));q=[float(sum(w*abs(v-target))) for v in pred]
                    close(counts,block['counts']);close(means,block['empirical_repeat']);close(tv,block['bin_mark_TV'])
                    close(l,old['metrics'][group]['metrics']['short_gap_repeat_curve_l1']);close(l,block['empirical_L1'])
                    for j,target_name in enumerate(c['models']):
                        close(pred[j],block['predicted_repeat'][target_name]);close(q[j],block['expected_curve_L1'][target_name])
                    close(arrays['mark_tv'][mask].mean(),block['mark_TV_transition_mean'])
                    empirical[(name,repeat,group)]=l;curves[(name,repeat,group)]=q;n+=1
        for r in c['repeats']:
            for g in c['groups']:
                uu,eu=curves[('Ucal',r,g)];ue,ee=curves[('Ecal',r,g)]
                d=out['decomposition'][str(r)][g]
                f=((eu-uu)+(ee-ue))/2;h=((ue-uu)+(ee-eu))/2
                native=empirical[('Ecal',r,g)]-empirical[('Ucal',r,g)];s=native-(ee-uu)
                for key,value in dict(predictor_F=f,source_H=h,empirical_remainder_S=s,expected_difference=ee-uu,native_difference=native).items():close(value,d[key])
                close(native,f+h+s);decompositions+=1
        print(f'verified pi={pi} kappa={k} trial={t}',flush=True)
    result=dict(status='PASS',source_commit=terminal['source_commit'],source_group_checks=n,
        expected_L1_values=2*n,native_L1_values=n,decomposition_checks=decompositions,max_absolute_error=maxerr,
        verifier_sha256=sha256(__import__('pathlib').Path(__file__)))
    write_json(OUTPUT/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
