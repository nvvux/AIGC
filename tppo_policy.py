# tppo_policy.py
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Chuẩn thiết kế cho lịch sử Stackelberg game:
    - Đầu vào: (batch, seq_len * input_dim) -- input_dim=2 (giá, số tham gia)
    - output: (batch, features_dim)
    """
    def __init__(self, observation_space, features_dim=64, nhead=4, nlayer=2, seq_len=10):
        super().__init__(observation_space, features_dim)
        self.seq_len = seq_len
        self.input_dim = observation_space.shape[0] // seq_len
        self.linear_emb = nn.Linear(self.input_dim, features_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim, nhead=nhead, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
        # Không positional encoding, đúng như thực nghiệm paper
    def forward(self, observations):
        # Chuẩn hóa tensor về float, đúng chiều batch
        # observations shape: (batch, seq_len * input_dim)
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.seq_len, self.input_dim).float()
        x = self.linear_emb(x)  # (batch, seq_len, features_dim)
        x = self.transformer(x)  # (batch, seq_len, features_dim)
        # Lấy vector của bước cuối (cuối chuỗi), chuẩn RL paper
        x = x[:, -1, :]  # (batch, features_dim)
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
                features_dim=features_dim,
                nhead=nhead,
                nlayer=nlayer,
                seq_len=seq_len
            ),
            **kwargs
        )
