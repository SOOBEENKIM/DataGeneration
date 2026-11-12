import numpy as np
import torch
from benchmarks.argn_transition_reference import probabilities, reference_table


def test_reference_uses_previous_state_and_completed_past_run_only():
    # Actual sequence 0, 1, 1, 0: the third event sees a run of ONE, not two.
    params={'previous_label':{'-1':.08,'0':.001,'1':.9},
            'age_hazard':{'(1, 1)':.95,'(1, 2)':.8}}
    features=torch.zeros(4,10)
    features[:,0]=torch.tensor([0.,1.,1.,1.])
    features[:,1]=torch.tensor([0.,0.,1.,1.])
    features[:,3]=torch.tensor(np.log1p([0,1,1,2])/np.log1p(3106))
    actual=probabilities(features,torch.tensor(reference_table(params,'duration')),3106)
    torch.testing.assert_close(actual,torch.tensor([.08,.001,.95,.8],dtype=torch.float64))
    # Current amount/category/label never enters this statistical reference.
    features[:,4:]=100
    torch.testing.assert_close(actual,probabilities(features,torch.tensor(reference_table(params,'duration')),3106))


def test_unseen_duration_falls_back_and_tail_age_is_pooled():
    params={'previous_label':{'-1':.08,'0':.001,'1':.9},'age_hazard':{'(1, 21)':.4}}
    f=torch.zeros(3,10);f[:,0:2]=1
    f[:,3]=torch.tensor(np.log1p([3,21,1000])/np.log1p(3106))
    torch.testing.assert_close(probabilities(f,torch.tensor(reference_table(params,'duration')),3106),
                               torch.tensor([.9,.4,.4],dtype=torch.float64))
    torch.testing.assert_close(probabilities(f,torch.tensor(reference_table(params,'markov')),3106),
                               torch.tensor([.9,.9,.9],dtype=torch.float64))
