import torch
from torch import nn

class RAAN(nn.Module):
    def __init__(self, num_features, attention, num_filters=3, input_size=1024, fc_output=256):
        super(RAAN, self).__init__()
        self.num_features = num_features
        self.attention = attention
        self.num_filters = num_filters
        self.input_size = input_size
        self.fc_output = fc_output

        self._prepare_raan_model()

    def _prepare_raan_model(self):
        self.att_net = nn.ModuleList()

        for i in range(0, self.num_filters):
            self.att_net.append(nn.Sequential(
                nn.Linear(self.input_size, self.fc_output),
                nn.ReLU(),
                nn.Linear(self.fc_output, 1),
                nn.Softmax(dim=1)))
        
        self.fc = nn.Linear(self.input_size, 1)

    def forward(self, input):
        input = input.view(-1, self.num_features, self.input_size)
        if self.attention:
            att_list = []
            for i in range(0, self.num_filters):
                att_list.append(self.att_net[i](input))
            all_atts = torch.stack(att_list, 2)
            att = torch.mean(all_atts, 2)
        else:
            att = torch.ones((input.size(0),self.num_features, 1)).cuda() * (1.0/self.num_features)
        input = torch.mul(input, att)
        input = input.sum(1)
        input = self.fc(input).view(-1)
        return input, att
