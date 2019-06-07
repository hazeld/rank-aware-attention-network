import torch
from torch import nn

class RAAN(nn.Module):
    def __init__(self, num_features, input_size=1024):
        super(RAAN, self).__init__()
        self.input_size = input_size
        self.num_features = num_features

        self._prepare_raan_model()

    def _prepare_raan_model(self):
        self.fc = nn.Linear(self.input_size, 1)

    def forward(self, input):
        input = input.view(-1, self.num_features, self.input_size)
        att = torch.ones((input.size(0),self.num_features, 1)).cuda() * (1.0/self.num_features)
        input = torch.mul(input, att)
        input = input.sum(1)
        input = self.fc(input).view(-1)
        return input, att
