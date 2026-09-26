"""Freeze common support from static inputs before CPAR generation."""
from run_sparkov_argn_control import OUT,DOCS,pd,write,digest

base=OUT/'prepared';train=pd.read_parquet(base/'train_context.parquet');val=pd.read_parquet(base/'validation_context.parquet')
supported=pd.Series(True,index=val.index);unknown={}
for field in ['cardholder_gender','cardholder_state']:
    levels=set(train[field]);supported &=val[field].isin(levels)
    unknown[field]=sorted(set(val[field])-levels)
path=base/'supported_validation_context.parquet';assert not path.exists()
val[supported].to_parquet(path,index=False)
write(DOCS/'context_support.json',dict(requested_customers=len(val),supported_customers=int(supported.sum()),unsupported_customers=int((~supported).sum()),unknown_levels=unknown,selection_uses_outcomes=False,original_argn_outputs_retained=True,supported_context_sha256=digest(path),protocol_sha256=digest(DOCS/'CPAR_CONTEXT_SUPPORT_PROTOCOL.md')))
