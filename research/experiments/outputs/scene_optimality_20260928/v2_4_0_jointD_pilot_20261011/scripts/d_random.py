import numpy as np
import torch
from hrl_v2.policy import Policy
from hrl_v2.randomness import PURPOSES
if 'random_upper' not in PURPOSES:
    assert max(PURPOSES.values()) == 12
    PURPOSES['random_upper'] = 13
assert PURPOSES['random_upper'] == 13

class RandomUpperPolicy(Policy):
 def __init__(self,*args,random_seed,**kwargs):
  super().__init__(*args,**kwargs);self.random_upper=np.random.default_rng(random_seed);self.vectors=[]
 @torch.no_grad()
 def upper(self,report,history):
  if self.observer is not None:self.observer.update(report,history)
  u=self.random_upper.uniform(-1,1,6);self.vectors.append(u.copy());return u,{}
