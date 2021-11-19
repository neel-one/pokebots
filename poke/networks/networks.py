import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['Network', 'LSTM']


class LSTM(nn.Module):
    pass


class Network(nn.Module):
    in_size = 110
    out_size = 13

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        assert (self.in_size,) == input_shape
        assert (self.out_size,) == output_shape

        self.h1 = nn.Linear(self.in_size, 150)
        self.h2 = nn.Linear(150, 70)
        self.h3 = nn.Linear(70, 30)
        self.h4 = nn.Linear(30, self.out_size)

        nn.init.xavier_uniform_(self.h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.h4.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        qs = self.h4(x)
        if action is None:
            return qs
        else:
            q_acted = torch.squeeze(qs.gather(1, action.long()))
            return q_acted

