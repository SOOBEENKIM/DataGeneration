"""Minimal gap-only expert for observed/generated label transitions."""
from contextlib import contextmanager
import torch
from benchmarks.argn_amount_control import AmountHeads
from benchmarks.argn_onset_output import onset_output_generation


def boundary_mask(features,labels):
    return features[...,0].gt(.5)&features[...,2].le(.5)&features[...,1].gt(.5).ne(labels.bool())


@contextmanager
def boundary_gap_generation(*args,boundary_payload,**kwargs):
    import mostlyai.engine._tabular.generation as generation
    with onset_output_generation(*args,**kwargs):
        parent=generation.SequentialModel
        class BoundaryModel(parent):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                payload=torch.load(boundary_payload,map_location=self.device,weights_only=True)
                devices=[self.device.index or 0] if self.device.type=='cuda' else []
                with torch.random.fork_rng(devices=devices):head=AmountHeads(self,payload['keys']).to(self.device)
                head.load_state_dict(payload['state_dict']);head.eval().requires_grad_(False)
                object.__setattr__(self,'boundary_head',head);self.boundary_inputs={}
                self.regressors.register_forward_pre_hook(self.capture_boundary)
                self.predictors.register_forward_hook(self.replace_boundary)

            def capture_boundary(self,module,args):
                if args[1] in self.boundary_head.keys:self.boundary_inputs[args[1]]=torch.cat(args[0],-1)

            def replace_boundary(self,module,args,output):
                if args[1] not in self.boundary_head.keys:return output
                x=self.boundary_inputs.pop(args[1]);mask=boundary_mask(self._state_features,self._amount_labels)
                value=self.boundary_head(x,args[1],self._amount_labels)
                return torch.where(mask.unsqueeze(-1),value,output)
        generation.SequentialModel=BoundaryModel
        try:yield
        finally:generation.SequentialModel=parent
