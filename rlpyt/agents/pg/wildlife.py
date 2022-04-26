from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.models.pg.wildlife_ff_model import WildlifeGraphModel, \
    WildlifeBasisGraphModel


class WildlifeMixin:

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


class WildlifeGraphAgent(WildlifeMixin, CategoricalPgAgent):
    """Creates agent for regular GNN model."""
    def __init__(self, ModelCls=WildlifeGraphModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class WildlifeBasisGraphAgent(WildlifeMixin, CategoricalPgAgent):
    """Creates agent for MMDP homomorphic GNN model."""
    def __init__(self, ModelCls=WildlifeBasisGraphModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
