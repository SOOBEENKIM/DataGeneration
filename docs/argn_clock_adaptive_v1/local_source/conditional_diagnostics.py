"""Explain frozen fitted corrections; never optimize or open development events."""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from run_adaptive import *


def main():
    verify();old=payload()
    limits=json.loads((DOCS/'CLOCK_ENDPOINTS.json').read_text())['values']
    prior=json.loads((ROOT/'docs/argn_clock_location_v1/clock_prefix_fit.json').read_text())['selected']
    parameters={'clock_joint':[0.,0.], 'clock_prefix':[prior['a'],prior['b']]}
    parameters.update({arm:json.loads((DOCS/f'{arm}_fit.json').read_text())['selected']['theta'] for arm in ARMS})
    rows=[];bins=[(1,50),(50,200),(200,500),(500,1024),(1024,2048),(2048,np.inf)]
    for split,seeds in [('optimization',[CFG['fit_noise_seed']]),('internal_validation',CFG['diagnostic_seeds'])]:
        meta=pd.read_parquet(META/f'metadata_{split}.parquet');meta=meta[meta.event_index.gt(0)]
        for noise in seeds:
            sim=Simulation(split,old,noise,'event')
            clock_bin=np.digitize(sim.x[:,0],[limits[0],sum(limits)/2,limits[1]])
            for arm,theta in parameters.items():
                delta=np.clip(basis_numpy(arm if arm in ARMS else 'event_two',sim.x,sim.valid,limits)@theta,-LIMIT,LIMIT)
                generated=np.floor(np.expm1(np.clip(sim.base_z+delta*sim.nn,0,np.log1p(1045691.)))+1e-8)
                for low,high in bins:
                    for clock_group in [-1,0,1,2,3]:
                        mask=sim.nn&meta.event_index.ge(low).to_numpy()&meta.event_index.lt(high).to_numpy()
                        if clock_group!=-1:mask&=(clock_bin==clock_group)&(sim.valid>.5)
                        if not mask.any():continue
                        x=np.log1p(meta.gap.to_numpy()[mask]);y=np.log1p(generated[mask])
                        weights=equal_weights(sim.ids[mask])
                        rows.append(dict(split=split,noise_seed=noise,arm=arm,start=low,
                            end=-1 if not np.isfinite(high) else int(high),clock_group=clock_group,
                            events=int(mask.sum()),customers=len(np.unique(sim.ids[mask])),
                            event_w1=wasserstein_distance(x,y),
                            customer_w1=wasserstein_distance(x,y,u_weights=weights,v_weights=weights),
                            mean_delta=float(delta[mask].mean())))
    df=pd.DataFrame(rows);df.to_csv(DOCS/'conditional_positions_and_clocks.csv',index=False)
    means=df[df.clock_group.eq(-1)].groupby(['split','arm','start'])[['events','customers','event_w1','customer_w1','mean_delta']].mean()
    means.to_csv(DOCS/'conditional_position_means.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    n=np.arange(1,3106)
    colors={'clock_prefix':'#555555','event_two':'#b88524','customer_two':'#22936c','position_six':'#337fc1'}
    for arm,color in colors.items():
        x=np.c_[np.full(len(n),sum(limits)/2),np.log1p(n)]
        y=np.clip(basis_numpy(arm if arm in ARMS else 'event_two',x,np.ones(len(n)),limits)@parameters[arm],-LIMIT,LIMIT)
        axes[0].plot(n,y,label=arm,color=color)
    for c,label,color in [(limits[0],'Fast past clock (q10)','#3679b0'),(sum(limits)/2,'Middle past clock','#627769'),(limits[1],'Slow past clock (q90)','#b6614b')]:
        x=np.c_[np.full(len(n),c),np.log1p(n)]
        y=basis_numpy('personal_six',x,np.ones(len(n)),limits)@parameters['personal_six']
        axes[1].plot(n,y,label=label,color=color)
    for ax in axes:
        ax.axhline(0,color='black',linewidth=.6);ax.set(xlabel='Number of previous transactions',ylabel='Normal-state log(1 + gap) correction')
        ax.grid(alpha=.15);ax.legend(fontsize=8)
    axes[0].set_title('Weighting and position controls');axes[1].set_title('Same six coefficients, position × personal clock')
    for suffix in ['png','pdf']:fig.savefig(DOCS/f'learned_corrections.{suffix}',dpi=180)
    plt.close(fig)
    write(DOCS/'CONDITIONAL_DIAGNOSTIC_COMPLETE.json',dict(source_sha256=digest(__file__),plan_sha256=digest(DOCS/'CONDITIONAL_DIAGNOSTIC_PLAN.md'),rows=len(df),new_fits=0,test_events_read=False,development_events_read=False))
    verify();print('CONDITIONAL_DIAGNOSTICS_COMPLETE',len(df),flush=True)


if __name__=='__main__':main()
