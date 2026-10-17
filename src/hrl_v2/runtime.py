"""Explicit production device contract and independently seeded learners."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
import hashlib
import json
from .randomness import derive,initialization
from .networks import LowerActor,QNetwork,UpperActor,ValueNetwork
from .sac import SAC
from .ppo import PPO


def configure(*,test_cpu=False):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":
        raise RuntimeError("V2 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.set_num_threads(10)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    if test_cpu: return torch.device("cpu")
    if not torch.cuda.is_available(): raise RuntimeError("Protocol requires cuda:0; no automatic fallback")
    return torch.device("cuda:0")


def parameter_hash(module):
    h=hashlib.sha256()
    for key,value in module.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Learner:
    def __init__(self,seed,device,config=None):
        if config is None:
            from .paths import default_config
            config=json.loads(default_config().read_text(encoding="utf-8"))
        kind=config.get("encoder","graph")
        if kind=="lite":
            from .lite_networks import LiteLowerActor,LiteQ,LiteUpperActor,LiteValue
            classes=(LiteLowerActor,LiteQ,LiteQ,LiteUpperActor,LiteValue)
        elif kind=="graph": classes=(LowerActor,QNetwork,QNetwork,UpperActor,ValueNetwork)
        else: raise ValueError("Unknown observation encoder")
        networks=[]
        for purpose,cls in zip(("actor","q1","q2","upper_init","value"),classes):
            with initialization(derive(seed,purpose=purpose)): network=cls()
            networks.append(network.to(device))
        lower,q1,q2,upper,value=networks
        self.sac=SAC(lower,q1,q2,gamma=config["objective"]["gamma_micro"])
        self.ppo=PPO(upper,value,gamma=config["ppo"]["gamma"],gae_lambda=config["ppo"]["gae_lambda"])

    def state_dict(self):
        result={}
        for prefix,obj,names in (("sac",self.sac,("actor","q1","q2","target1","target2","actor_optimizer","critic_optimizer")),
                                  ("ppo",self.ppo,("actor","value","actor_optimizer","value_optimizer"))):
            result[prefix]={name:getattr(obj,name).state_dict() for name in names}
            result[prefix].update(counts=dict(obj.counts),version=obj.version)
        result["discounts"]=dict(sac=self.sac.gamma,ppo=self.ppo.gamma,gae_lambda=self.ppo.gae_lambda)
        return result

    def load_state_dict(self,state):
        expected=dict(sac=self.sac.gamma,ppo=self.ppo.gamma,gae_lambda=self.ppo.gae_lambda)
        if state.get("discounts")!=expected:
            raise ValueError("Checkpoint discounts differ from the explicit learner configuration")
        for prefix in ("sac","ppo"):
            obj=getattr(self,prefix)
            for name,value in state[prefix].items():
                if name in ("counts","version"): setattr(obj,name,value)
                else: getattr(obj,name).load_state_dict(value)
        for network in (self.sac.actor,self.sac.q1,self.sac.q2,self.sac.target1,self.sac.target2,self.ppo.actor,self.ppo.value):
            for module in network.modules():
                if isinstance(module,torch.nn.GRU): module.flatten_parameters()
