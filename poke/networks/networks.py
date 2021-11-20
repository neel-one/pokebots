import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

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
        
        # Ideally 

        self.h1 = nn.Linear(self.in_size, 150)
        self.h2 = nn.Linear(150, 70)
        var_depth = []
        n_depth = 4
        for i in range(n_depth):
            var_depth.extend([nn.Linear(70,70), nn.ReLU()])
        self.h3 = nn.Sequential(*var_depth)
        self.h4 = nn.Linear(70, self.out_size)

        nn.init.xavier_uniform_(self.h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.h3.weight,
        #                         gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.h4.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        if torch.cuda.is_available():
            state = state.cuda()
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        qs = self.h4(x)
        if action is None:
            return qs
        else:
            q_acted = torch.squeeze(qs.gather(1, action.long()))
            return q_acted

if __name__ == '__main__':
    summary(Network((110,),(13,)), input_size=(32,110))