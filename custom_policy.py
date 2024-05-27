from hflayers import Hopfield
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class HopfieldFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, num_heads=1, scaling=1.0):
        super(HopfieldFeaturesExtractor, self).__init__(observation_space, features_dim)
        # Include the Hopfield layer
        self.hopfield_layer = Hopfield(
            input_size=observation_space.shape[0],
            hidden_size=features_dim,
            output_size=features_dim,
            num_heads=num_heads,
            scaling=scaling,
            batch_first=True,

            # do not pre-process layer input
            normalize_stored_pattern=False,
            normalize_stored_pattern_affine=False,
            normalize_state_pattern=False,
            normalize_state_pattern_affine=False,
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,
            # do not post-process layer output
            disable_out_projection=True
        )

    def forward(self, observations):
        observations = observations.unsqueeze(1)
        x = self.hopfield_layer(observations)
        # Flatten the output for compatibility with SB3 policies
        return x.view(x.size(0), -1)


class CustomHopfieldPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomHopfieldPolicy, self).__init__(*args, **kwargs,
                                                   features_extractor_class=HopfieldFeaturesExtractor)
