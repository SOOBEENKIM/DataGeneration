"""Adaptive, registered order sensitivity with frozen ARGN weights."""
import argparse
import inspect
import shutil
import time
from run_sparkov_argn_control import OUT,DOCS,CFG,LABEL,pd,seed,write,digest,native_frame
from mostlyai.engine import generate
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_argn_name,SLEN_SIDX_SDEC_COLUMN
import mostlyai.engine._tabular.generation as module

def run(fit_seed,source_arm='relaxed',priorities=None):
    origin=OUT/'runs'/f'{source_arm}_{fit_seed}';assert (origin/'FIT.json').exists()
    original=module._resolve_gen_column_order
    for priority in priorities or ['category',LABEL]:
        folder=OUT/'runs'/f'order_{priority}_{source_arm}_{fit_seed}';folder.mkdir(parents=True,exist_ok=False)
        wsdir=folder/'workspace';shutil.copytree(origin/'workspace',wsdir,ignore=shutil.ignore_patterns('SyntheticData'))
        ws=Workspace(wsdir);weights=digest(ws.model_tabular_weights_path);st=ws.tgt_stats.read()['columns'][priority]
        column=get_argn_name(st['argn_processor'],st['argn_table'],st['argn_column'])
        def order(**kwargs):
            native=original(**kwargs);assert native[0]==SLEN_SIDX_SDEC_COLUMN and column in native
            return [native[0],column]+[c for c in native if c not in [native[0],column]]
        extra=DOCS/'AMOUNT_SCALE_PROTOCOL.md' if source_arm=='numeric_digit_relaxed' else DOCS/'DIGIT_FIT_PROTOCOL.md' if source_arm=='digit_relaxed' else None
        write(folder/'FIT.json',dict(frozen_checkpoint_source=str(origin),weights_sha256=weights,priority=priority,protocol_sha256=digest(DOCS/'GENERATION_ORDER_PROTOCOL.md'),encoding_control_protocol_sha256=digest(extra) if extra else None,script_sha256=digest(__file__),refit=False))
        module._resolve_gen_column_order=order
        try:
            ctx=pd.read_parquet(OUT/'prepared/validation_context.parquet')
            for gs in CFG['generation_seeds']:
                seed(gs);start=time.monotonic();generate(ctx_data=ctx,device='cpu',workspace_dir=wsdir)
                raw=pd.read_parquet(wsdir/'SyntheticData');raw.to_parquet(folder/f'native_validation_{gs}.parquet',index=False)
                d=native_frame(raw);dest=folder/f'generated_validation_{gs}.parquet';d.to_parquet(dest,index=False)
                assert digest(ws.model_tabular_weights_path)==weights
                write(folder/f'generation_validation_{gs}.json',dict(seconds=time.monotonic()-start,events=len(d),customers=d.entity_id.nunique(),weights_unchanged=True,sha256=digest(dest)))
                print(f'ORDER {fit_seed} {priority} {gs}: {len(d)}',flush=True)
        finally:module._resolve_gen_column_order=original

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fit_seed',type=int);p.add_argument('--source-arm',default='relaxed',choices=['relaxed','digit_relaxed','numeric_digit_relaxed']);p.add_argument('--priority',choices=['category',LABEL]);args=p.parse_args()
    run(args.fit_seed,args.source_arm,[args.priority] if args.priority else None)
