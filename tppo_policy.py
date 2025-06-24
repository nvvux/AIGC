import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, nhead=4, nlayer=2, seq_len=10):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0] // seq_len
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.linear_emb = nn.Linear(input_dim, features_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=features_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
    def forward(self, observations):
        batch_size = observations.size(0)
        x = observations.view(batch_size, self.seq_len, self.input_dim)
        x = self.linear_emb(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x[-1]  # [batch, features_dim]
        return x

class TransformerActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        seq_len = kwargs.pop("seq_len", 10)
        features_dim = kwargs.pop("features_dim", 64)
        nhead = kwargs.pop("nhead", 4)
        nlayer = kwargs.pop("nlayer", 2)
        super().__init__(
            *args,
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=features_dim, nhead=nhead, nlayer=nlayer, seq_len=seq_len
            ),
            **kwargs
        )

