import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=12,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
        )
        self.fc1 = nn.Linear(in_features=32, out_features=12)

    def forward(self, x):
        output, (hidden, _) = self.lstm(x)
        out = self.fc1(hidden[-1])
        return out
