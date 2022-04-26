
from collections import namedtuple

from rlpyt.utils.collections import namedarraytuple, AttrDict


Samples = namedarraytuple("Samples", ["agent", "env"])

AgentSamples = namedarraytuple("AgentSamples",
    ["action", "prev_action", "agent_info"])
AgentSamplesBsv = namedarraytuple("AgentSamplesBsv",
    ["action", "prev_action", "agent_info", "bootstrap_value"])
EnvSamples = namedarraytuple("EnvSamples",
    ["observation", "reward", "prev_reward", "done", "env_info"])


class BatchSpec(namedtuple("BatchSpec", "T B")):
    """
    T: int  Number of time steps, >=1.
    B: int  Number of separate trajectory segments (i.e. # env instances), >=1.
    """
    __slots__ = ()

    @property
    def size(self):
        return self.T * self.B


class TrajInfo(AttrDict):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0.0
        self.Return = 0
        self.NonzeroRewards = 0
        self.DiscountedReturn = 0
        self._cur_discount = 1
        self.scapt = 0.0
        self.fcapt = 0.0
        self.colliders = 0.0
        self.wait_time = 0.0
        self.sim_time = 0.0

    def step(self, observation, action, reward, done, agent_info, env_info):
        self.Length += 1.0
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.DiscountedReturn += self._cur_discount * reward
        self._cur_discount *= self._discount
        try:
            self.scapt += env_info.scapt
            self.fcapt += env_info.fcapt
            self.colliders += env_info.colliders
        except AttributeError:
            pass
        try:
            self.wait_time += env_info.wait_time
            self.sim_time += env_info.sim_time
        except AttributeError:
            pass

    def terminate(self, observation):
        self.wait_time = self.wait_time/self.Length
        #print("terminating from", self.scapt, self.fcapt, self.colliders,
        #      self.Return, self.Length)
        return self
