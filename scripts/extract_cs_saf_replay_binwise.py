"""Export every registered binwise calibration contrast from complete evidence."""
from pathlib import Path
import json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_cs_saf_replay_oracle import stats
from experiments.cs_saf_replay_oracle import contract
from scripts.audit_cs_saf_oracle import sha256

def main():
    c=contract(); root=ROOT/c['output']; full=json.loads((root/'summary.json').read_text())
    replays={(r['pi'],r['kappa'],r['trial']):r for r in full['raw_replays']}
    oracles={(r['pi'],r['kappa'],r['trial']):r for r in full['raw_oracle_reports']}
    result={'source_commit':full['execution']['source_commit'],'full_summary_sha256':sha256(root/'summary.json'),
        'aggregation':'Three tapes averaged within trial, then five trial summaries. Bias = probability minus full-validation observed repeat in fixed train metric bin. Not new fits or a new endpoint.', 'cells':{}}
    for pi in c['prevalences']:
      for k in c['kappas']:
        cellkey=f'pi_{pi:.2f}/kappa_{k}'; result['cells'][cellkey]={}
        for label in ('0','1'):
          example=replays[pi,k,0]; output={'metric_edges':example['metric_edges'][label],
            'raw_reference':example['reference'][label],'quantized_reference':example['quantized_reference'][label], 'replay':{}, 'oracle':{}}
          for mode in c['replay_modes']:
            for target in c['models']:
              for source in c['models']:
                prefix=f'{mode}/target_{target}/source_{source}'; groups=[]
                for trial in c['trials']:
                  entries=[v[label] for key,v in replays[pi,k,trial]['matrices'].items() if key.startswith(mode+'/') and key.endswith(f'/target_{target}/source_{source}')]
                  groups.append(np.mean([e['model_repeat'] for e in entries],axis=0))
                values=np.array(groups); reference=np.array(example['reference'][label]['observed_repeat'])
                output['replay'][prefix]={'predicted_repeat':[stats(values[:,b]) for b in range(5)],
                  'signed_bias':[stats(values[:,b]-reference[b]) for b in range(5)]}
          for mode in c['oracle_modes']:
            for refname in ('raw_reference','matched_reference'):
              groups=[]
              for trial in c['trials']:
                entries=[v[refname][label] for key,v in oracles[pi,k,trial]['controls'].items() if key.startswith(mode+'_')]
                groups.append(np.mean([e['model_repeat'] for e in entries],axis=0))
              reference=np.array(example['quantized_reference' if refname=='matched_reference' and mode.endswith('BIN') else 'reference'][label]['observed_repeat'])
              values=np.array(groups)
              output['oracle'][mode+'/'+refname]={'predicted_repeat':[stats(values[:,b]) for b in range(5)],
                'signed_bias':[stats(values[:,b]-reference[b]) for b in range(5)]}
          result['cells'][cellkey][label]=output
    path=root/'binwise_result.json'; path.write_text(json.dumps(result,indent=2)+'\n')
    print(str(path),sha256(path))
if __name__=='__main__': main()
