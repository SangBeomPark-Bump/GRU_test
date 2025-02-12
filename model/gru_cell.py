import torch
import torch.nn as nn

class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=True)

    def forward(self, x, h):
        x_gates = self.x2h(x)
        h_gates = self.h2h(h)
        
        x_r, x_z, x_n = x_gates.chunk(3, 1)
        h_r, h_z, h_n = h_gates.chunk(3, 1)

        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        n = torch.tanh(x_n + r * h_n)
        h_new = (1 - z) * n + z * h
        return h_new