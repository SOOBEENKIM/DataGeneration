"""Known phase-specific expert control; only normal-to-fraud amount output changes."""
from contextlib import contextmanager
import torch
from benchmarks.argn_amount_control import AmountHeads
from benchmarks.argn_joint_preservation import preservation_generation


def onset_mask(features,labels):
    return features[...,0].gt(.5)&features[...,1].le(.5)&labels.eq(1)


@contextmanager
def onset_output_generation(*args,onset_payload,**kwargs):
    import mostlyai.engine._tabular.generation as generation
    with preservation_generation(*args,**kwargs):
        parent=generation.SequentialModel
        class OnsetModel(parent):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                payload=torch.load(onset_payload,map_location=self.device,weights_only=True)
                devices=[self.device.index or 0] if self.device.type=='cuda' else []
                with torch.random.fork_rng(devices=devices):head=AmountHeads(self,payload['keys']).to(self.device)
                head.load_state_dict(payload['state_dict']);head.eval().requires_grad_(False)
                object.__setattr__(self,'onset_head',head);self.onset_inputs={}
                self.regressors.register_forward_pre_hook(self.capture_onset)
                self.predictors.register_forward_hook(self.replace_onset_amount)

            def capture_onset(self,module,args):
                if args[1] in self.onset_head.keys:self.onset_inputs[args[1]]=torch.cat(args[0],-1)

            def replace_onset_amount(self,module,args,output):
                if args[1] not in self.onset_head.keys:return output
                x=self.onset_inputs.pop(args[1]);mask=onset_mask(self._state_features,self._amount_labels)
                proposed=self.onset_head(x,args[1],self._amount_labels)
                return torch.where(mask.unsqueeze(-1),proposed,output)
        generation.SequentialModel=OnsetModel
        try:yield
        finally:generation.SequentialModel=parent
