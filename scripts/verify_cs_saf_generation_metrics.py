"""All 13 cached metrics must match immutable historical scores before dispatch."""
import json,time
import numpy as np
import torch
from experiments.cs_saf_generation_repeats import (
    ROOT,OUTPUT,CONFIG_SHA,contract,parent,ucal,ecal,frozen_source,
    canonical_and_plan,sample_to_frame,CachedGenerationMetrics,write_json,finish)


def main():
    c=contract();source=frozen_source();out=OUTPUT/'metric_gate'
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    records=[];largest=0.;started=time.monotonic()
    for pi in c['prevalences']:
        for k in c['kappas']:
            dataset,payload,plan,positions=canonical_and_plan(pi,k,0)
            evaluator=CachedGenerationMetrics(dataset,plan)
            for name in c['models']:
                if name.endswith('cal'):
                    module=ucal if name=='Ucal' else ecal
                    folder=module.OUTPUT/f'pi_{pi:.2f}/kappa_{k}/trial_0/{name}'
                    module.verify(folder);comparison=folder/'comparison.json'
                else:
                    folder=parent.folder_for(pi,k,0,name);parent.verify_artifacts(folder)
                    generation=parent.OUTPUT/f'generation/pi_{pi:.2f}/trial_0/kappa_{k}/{name}'
                    parent.verify_artifacts(generation);comparison=generation/'comparison.json'
                sample=torch.load(folder/'generated_sample.pt',map_location='cpu')['sample']
                actual=evaluator.score(sample_to_frame(sample,payload,plan,positions))
                expected=json.loads(comparison.read_text())['metrics']
                for group in actual:
                    if json.loads(json.dumps(actual[group]['train_metric_state']))!=expected[group]['train_metric_state']:
                        raise ValueError('train-fitted metric state changed')
                    if set(actual[group]['metrics'])!=set(expected[group]['metrics']):raise ValueError('metric suite changed')
                    for metric,value in actual[group]['metrics'].items():
                        error=abs(value-expected[group]['metrics'][metric]);largest=max(largest,error)
                        if error>1e-12:raise ValueError(f'cached arithmetic differs: {pi} {k} {name} {group} {metric} {error}')
                        records.append(dict(pi=pi,kappa=k,model=name,group=group,metric=metric,error=error))
            print(f'metric equivalence PASS pi={pi} kappa={k}',flush=True)
    write_json(out/'gate.json',dict(decision='PASS',source_commit=source,config_sha256=CONFIG_SHA,
        metrics_checked=len(records),max_absolute_error=largest,seconds=time.monotonic()-started,records=records))
    finish(out,source)


if __name__=='__main__':main()
