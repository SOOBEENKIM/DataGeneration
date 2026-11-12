"""Post-fit diagnostic only: decoding vs denoising vs free-prefix behavior."""
import sys,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
from run_lbdtpp import OUT,CFG,BDTPPModel,load_data,batches,collate,verify,digest,write

@torch.no_grad()
def main():
    verify();assert torch.cuda.is_available();records=[]
    for arm in ['official_scale','median_scale']:
        run=OUT/arm;ck=torch.load(run/'best.pt',map_location='cuda',weights_only=False)
        model=BDTPPModel(SimpleNamespace(**ck['args']),28).cuda();model.load_state_dict(ck['state_dict']);model.eval()
        data=load_data(ck['categories'],ck['scale'],'internal_validation')
        for setting in ['direct_latent','noise_0','noise_50','noise_99']:
            torch.manual_seed(20261430);torch.cuda.manual_seed_all(20261430)
            sums={k:dict(events=0,fraud=0,predicted_fraud=0,mark_correct=0,label_correct=0,
                         absolute_log_gap_error=0.,fraud_correct=0) for k in ['first8','8_63','64plus']}
            for ix in batches(data,False,None):
                x,e=collate(data,ix,28);valid=e.ne(28);latent=model.temporal_enc(x)+model.event_emb(e)
                latent=latent.masked_fill(~valid[:,:,None],0)
                if setting=='direct_latent':decoded=latent
                else:
                    time=int(setting.split('_')[1]);t=torch.full((len(ix),x.shape[1]//8),time,device='cuda',dtype=torch.long)
                    noisy=model.diffusion_model.q_sample(latent,t).masked_fill(~valid[:,:,None],0)
                    model.block_dit.remove_kv_cache();decoded=model.block_dit(torch.cat([noisy,latent],1))
                mark=model.type_decoder(decoded).argmax(-1)
                gap=model.time_decoder(decoded).squeeze(-1)*ck['scale']
                err=(torch.log1p(gap)-torch.log1p(x*ck['scale'])).abs()
                positions=torch.arange(x.shape[1],device='cuda')[None,:]
                for name,lo,hi in [('first8',1,8),('8_63',8,64),('64plus',64,10**9)]:
                    keep=valid&positions.ge(lo)&positions.lt(hi);s=sums[name]
                    truth=e.remainder(2).eq(1);pred=mark.remainder(2).eq(1)
                    s['events']+=int(keep.sum());s['fraud']+=int((keep&truth).sum())
                    s['predicted_fraud']+=int((keep&pred).sum())
                    s['mark_correct']+=int((keep&mark.eq(e)).sum());s['label_correct']+=int((keep&pred.eq(truth)).sum())
                    s['fraud_correct']+=int((keep&pred&truth).sum());s['absolute_log_gap_error']+=float(err[keep].sum())
            for group,s in sums.items():
                records.append(dict(arm=arm,setting=setting,group=group,**s,
                    true_fraud_rate=s['fraud']/s['events'],predicted_fraud_rate=s['predicted_fraud']/s['events'],
                    label_accuracy=s['label_correct']/s['events'],mark_accuracy=s['mark_correct']/s['events'],
                    fraud_recall=s['fraud_correct']/s['fraud'] if s['fraud'] else np.nan,
                    log_gap_mae=s['absolute_log_gap_error']/s['events']))
        assert digest(run/'best.pt')==json.loads((run/'FIT_COMPLETE.json').read_text())['sha256']
        del model;torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(OUT/'report/FROZEN_DECODER_DENOISER_DIAGNOSTIC.csv',index=False)
    write(OUT/'report/FROZEN_DIAGNOSTIC_COMPLETE.json',dict(new_fits=0,new_full_generations=0,
        checkpoint_selection_changed=False,test_events_read=False,
        qualification='noise probes use actual other-block history, not free-generation performance'))
    print('FROZEN_DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
