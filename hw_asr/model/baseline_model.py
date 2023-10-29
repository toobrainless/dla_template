from torch import nn
from torch.nn import Sequential

from .base_model import BaseModel


class BaselineModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.net = Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class),
        )

    def forward(self, spectrogram, **batch):
        input = spectrogram.transpose(1, 2)
        output = {"logits": self.net(input)}
        return output

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
