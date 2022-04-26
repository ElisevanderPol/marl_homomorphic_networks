from rlpyt.agents.pg.wildlife import WildlifeGraphAgent
from rlpyt.agents.pg.wildlife import WildlifeBasisGraphAgent
from rlpyt.agents.pg.traffic import TrafficGraphAgent
from rlpyt.agents.pg.traffic import TrafficBasisGraphAgent


def get_agent_cls_wildlife(agent_type, algo="ppo"):
    """
    Get agent wrapper for drone env.
    """
    if agent_type == "graph":
        return WildlifeGraphAgent, None
    elif agent_type == "eqgraph":
        return WildlifeBasisGraphAgent, agent_type
    else:
        raise TypeError("No agent of type {agent_type} known")


def get_agent_cls_traffic(agent_type, algo="ppo"):
    """
    Get agent wrapper for traffic env.
    """
    if agent_type == "graph":
        return TrafficGraphAgent, None
    elif agent_type == "eqgraph":
        return TrafficBasisGraphAgent, agent_type
    else:
        raise TypeError("No agent of type {agent_type} known")
