"""Read-only author-code preflight, not a paper-result reproduction or training."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'artifacts/reproduction_sources/vaegan_cpac_20260927'
DEST = ROOT / 'docs/sparkov_reference_reset_20260927'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    entry = AUTHOR / 'scripts/vaegan_cpac_fraud_run.py'
    tree = ast.parse(entry.read_text())
    definitions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    constants = {}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try: constants[node.targets[0].id]=ast.literal_eval(node.value)
            except (ValueError,TypeError): pass
    plot = definitions['save_latent_plot']
    formal = [a.arg for a in plot.args.args]
    plot_calls = [n for n in ast.walk(definitions['main']) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Name) and n.func.id=='save_latent_plot']
    unexpected = [kw.arg for call in plot_calls for kw in call.keywords if kw.arg not in formal]
    sig=inspect.Signature([inspect.Parameter(k,inspect.Parameter.POSITIONAL_OR_KEYWORD) for k in formal])
    try:
        sig.bind(*([None]*len(formal)),max_points_per_class=2000)
    except TypeError as exc:
        signature_failure=str(exc)
    main_calls = {n.func.id:n.lineno for n in ast.walk(definitions['main'])
                  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                  and n.func.id in ['normalize_all_features','stratified_train_val_split']}
    # Only read approved training amounts; no validation/test transactions.
    import numpy as np
    import pandas as pd
    import torch
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/reproduction_sources/matplotlib_cache'))
    sys.path.insert(0,str(AUTHOR))
    # Importing the large training module requires an unused plotting dependency.
    # Execute its exact standalone normalization definition, without altering it.
    norm_source=AUTHOR/'newDL_modules/training_funcs.py'
    norm_tree=ast.parse(norm_source.read_text())
    norm_def=next(n for n in norm_tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalize_all_features')
    namespace={}
    exec(compile(ast.Module(body=[norm_def],type_ignores=[]),str(norm_source),'exec'),namespace)
    normalize_all_features=namespace['normalize_all_features']
    from newDL_modules.cpac_vaegan_loop import generate_synthetic_frauds
    from newDL_modules.CPAClass import CausalPrototypeAttentionClassifier
    from deepLearning_modules.vae_ganClass import Encoder,Decoder
    source=ROOT.parent/'research-argn-fraud-audit/artifacts/sparkov_argn_control_v2/prepared/train.parquet'
    frame=pd.read_parquet(source,columns=['amount_or_numeric_value']).rename(columns={'amount_or_numeric_value':'amt'})
    normalized,_=normalize_all_features(frame,['amt'])
    torch.manual_seed(42); torch.set_num_threads(2)
    hidden=constants['VAE_HIDDEN_SIZE']; dim=16
    encoder=Encoder(dim,hidden).eval(); decoder=Decoder(dim,list(reversed(hidden))).eval()
    cpac=CausalPrototypeAttentionClassifier(input_dim=hidden[-1],hidden_dim=constants['CPAC_HIDDEN_DIM']).eval()
    with torch.no_grad():
        _,mu,_=encoder(torch.zeros(4,dim))
        probability=cpac(mu)
    x,y=generate_synthetic_frauds(decoder,cpac,num_samples=50,device='cpu')
    assert x.shape==(50,dim) and y.shape==(50,) and np.isfinite(x).all() and np.all(y==1)
    files=[entry,AUTHOR/'newDL_modules/cpac_vaegan_loop.py',AUTHOR/'newDL_modules/training_funcs.py',
           AUTHOR/'deepLearning_modules/vae_ganClass.py',AUTHOR/'README.md']
    result=dict(upstream='https://github.com/claudiunderthehood/VAEGAN-CPAC',
        commit=subprocess.check_output(['git','-C',str(AUTHOR),'rev-parse','HEAD'],text=True).strip(),
        code_hashes={str(p.relative_to(AUTHOR)):sha(p) for p in files},
        official_sparkov_entrypoint=str(entry.relative_to(AUTHOR)),
        declared_hyperparameters={k:constants[k] for k in ['RANDOM_STATE','VAL_SIZE','VAE_HIDDEN_SIZE','BATCH_SIZE','LR','VAE_MAX_EPOCHS','VAE_PATIENCE']},
        customer_id_dropped='cc_num' in constants['DROP_COLS'],
        generation_label='all ones assigned by author generation function; not a learned natural fraud prevalence',
        random_weight_interface_smoke={'passed':True,'generated_shape':list(x.shape),'training_performed':False,'quality_claim':False},
        preflight_runtime_note='Exact standalone normalization definition executed via AST; whole-module import required absent optional seaborn. This is not full author-environment reproduction.',
        plot_signature={'line':plot.lineno,'call_lines':[c.lineno for c in plot_calls],
                        'unsupported_keywords':unexpected,'reproduced_bind_error':signature_failure},
        validation_preprocessing_order=main_calls,
        normalization_precedes_validation_split=main_calls['normalize_all_features']<main_calls['stratified_train_val_split'],
        decoder_output_support=[0,1],
        permitted_training_amount={'rows':len(frame),'normalization_source':'unmodified author normalize_all_features',
             'fraction_outside_decoder_support':float(((normalized.amt<0)|(normalized.amt>1)).mean()),
             'normalized_min':float(normalized.amt.min()),'normalized_max':float(normalized.amt.max())},
        interpretation='Source compatibility checks only; no paper metric or fitted generator reproduced. No upstream fixes applied.',
        test_events_read=False)
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'CPAC_CODE_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
