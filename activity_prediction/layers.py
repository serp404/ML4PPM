from numpy.lib.arraysetops import union1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GruLstmLayer(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size,
        num_layers=1,
        bidirectional=False
    ):

        super(GruLstmLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional
        )

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional
        )

        self.output_states_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size * 2),
            nn.Dropout(p=0.25),
            nn.ELU(),
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size * 2),
            nn.ELU(),
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        )

    def forward(self, batch, init_states):
        h0, c0 = init_states
        _, (lstm_hn, _) = self.lstm(batch, (h0, c0))
        _, gru_hn = self.gru(batch, h0)
        
        union_lasts = torch.cat((lstm_hn[-1], gru_hn[-1]), dim=1)
        return self.output_states_layer(union_lasts)


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        
        # score function (simple MLP)
        self.score_function = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=1, bias=False),
            nn.Softmax(dim=1)
        )
    
    def forward(self, batch_prefix, batch_current):
        current_flattened = torch.unsqueeze(batch_current, dim=1).expand(batch_prefix.shape)

        scores = self.score_function(batch_prefix + current_flattened)
        weighted = batch_prefix * scores.expand(batch_prefix.shape)

        return torch.sum(weighted, dim=1, keepdim=False)
