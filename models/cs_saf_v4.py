"""The missing factorial arm: exact E forward, exact R residual penalty."""
from models.cs_saf_v3 import CSSAFv3

CANDIDATE = 'CS4-ER1'
VERSION = 'cs-saf-raw-forward-centered-penalty-v4'


class CSSAFv4(CSSAFv3):
    def __init__(self, support, reference_probabilities, **kwargs):
        # Keep the parent's E selector: changing it would select centered logits.
        super().__init__('CS3-E1', support, reference_probabilities, **kwargs)
        self.regularization_coefficient = .01

    def architecture_contract(self):
        return dict(super().architecture_contract(), implementation_version=VERSION,
                    candidate=CANDIDATE, prediction_implementation='CS3-E1',
                    centered=False, center_only_in_penalty=True)
