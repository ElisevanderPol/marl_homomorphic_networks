from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.models.pg.traffic_ff_model import TrafficGraphModel, \
    TrafficBasisGraphModel


class TrafficMixin:

    def make_env_to_model_kwargs(self, env_spaces):
        """Get args for model from environment."""
        im_shape = env_spaces.observation.shape
        im_shp = (env_spaces.action.n_agents,)+im_shape[1:]
        if env_spaces.action.decentralized:
            out = env_spaces.action.n_actions*env_spaces.action.n_agents
        else:
            out = env_spaces.action.n_actions**env_spaces.action.n_agents
        return dict(image_shape=im_shp,
                    output_size=out)


class TrafficGraphAgent(TrafficMixin, CategoricalPgAgent):
    """Create agent for regular GNN model."""
    def __init__(self, ModelCls=TrafficGraphModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class TrafficBasisGraphAgent(TrafficMixin, CategoricalPgAgent):
    """Create agent for MMDP homomorphic GNN model."""
    def __init__(self, ModelCls=TrafficBasisGraphModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
