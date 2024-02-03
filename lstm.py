import torch
from torch import nn
import math

class Loss(nn.Module):
    def __init__(self, r, p1):
        super(Loss, self).__init__()
        self.r = r
        self.p1 = p1

    def forward(self, final_out, point_victor):
        length = len(point_victor)
        loss = 0
        for i in range(length):
            ri = i / sum(range(0, length + 1))
            if point_victor[i] == [1,0] :
                loss += (math.log(final_out[i][0]) - self.r * (final_out[i][0] - self.p1)**2) * ri
            else:
                loss += (math.log(final_out[i][1]) - self.r * (final_out[i][0] - self.p1)**2) * ri

        loss = -loss

        return loss


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_cov = nn.Conv1d(input_size, hidden_size, kernel_size=1, padding=0, bias=False)
        self.line = nn.Linear(input_size, hidden_size)
        self.input_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=False)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)  # 新增的dropout层
        self.fc3 = nn.Linear(hidden_size, num_classes)  # 原来的输出层现在变成了第三个全连接层
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input):
        out = self.line(input.permute(1, 0))
        out, _ = self.input_lstm(out)
        out = self.fc3(out)
        #out = self.dropout(out)
        #out = self.fc3(out)
        out = self.softmax(out)

        return out

