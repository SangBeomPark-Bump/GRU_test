import torch
import torch.nn as nn
from gru_cell import MyGRUCell
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        self.gru_cell = MyGRUCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=torch.float)
        outputs = []
        for i in range(x.size(1)):
            h = self.gru_cell(x[:, i, :], h)
            outputs.append(h)
        
        output = outputs[-1]  # 마지막 hidden state 사용
        output = self.fc(output)
        output = self.sigmoid(output)
        return output, h