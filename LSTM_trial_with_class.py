import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim

classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]

# df = pd.read_csv("valid_random_strings.csv")
datalist = np.array(
    [
        "GTaACaHE",
        "GTaAC-1H1a1HE",
        "GTaACH-1H1a1HE",
        "GTa1bAC-2H2b2-1aT1HE",
    ]
)


def one_hot_encoding(datalist):
    one_hot_tensors = []
    for sequence in datalist:
        # Perform one-hot encoding for the sequence
        one_hot_encoded = []
        i = 0
        while i < len(sequence):
            char = sequence[i]
            vector = [0] * len(classes)  # Initialize with zeros

            if char == "-":
                next_char = sequence[i + 1]
                unit = char + next_char
                if unit in classes:
                    vector[classes.index(unit)] = 1
                    i += 1  # Skip the next character since it forms a unit
            elif char in classes:
                vector[classes.index(char)] = 1

            one_hot_encoded.append(vector)
            i += 1

        # Convert the list to a PyTorch tensor
        one_hot_tensor = torch.tensor(one_hot_encoded)
        one_hot_tensors.append(one_hot_tensor)

    return one_hot_tensors


def padding(one_hot_tensors):
    # Pad the one-hot tensors to have the same length
    padded_tensors = pad_sequence(
        one_hot_tensors, batch_first=True, padding_value=0
    ).float()

    return padded_tensors


class LSTMtry(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMtry, self).__init__()
        # self.lstm1 = nn.LSTM(input_size, hidden_size)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.dlayer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # out, hidden = self.lstm1(x)
        # out, _ = self.lstm2(out, hidden)
        out, _ = self.lstm(x)
        out = self.dlayer(out)

        return out


one_hot_tensors = one_hot_encoding(datalist)
padded_tensors = padding(one_hot_tensors)


# Define the LSTM model

model = LSTMtry(input_size=len(classes), hidden_size=32, num_classes=len(classes))

criterion = nn.CrossEntropyLoss()

# Define the optimizer
learning_rate = 0.001
optimizer = optim.Adam(
    model.parameters(),
    learning_rate,
)

# Training loop
num_epochs = 501
for epoch in range(num_epochs):
    model.train()
    # Forward pass
    output = model(padded_tensors)

    # Calculate the loss
    # breakpoint()
    loss = criterion(output.permute(0, 2, 1), torch.argmax(padded_tensors, axis=2))

    # # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # # Printing the loss
    if epoch % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        print(f"{torch.argmax(output, axis= 2)}")
print(torch.argmax(padded_tensors, axis=2))
