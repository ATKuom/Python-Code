import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import config

classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]


def gandetoken(datalist):
    for ind in range(len(datalist)):
        datalist[ind] = "G" + datalist[ind] + "E"
    return datalist


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


model = LSTMtry(input_size=len(classes), hidden_size=32, num_classes=len(classes))

criterion = nn.CrossEntropyLoss()

# Define the optimizer
learning_rate = 0.001
optimizer = optim.Adam(
    model.parameters(),
    learning_rate,
)

# 15% of the data is used for validation
if __name__ == "__main__":
    datalist = np.load(config.DATA_DIRECTORY / "D0.npy", allow_pickle=True)

    # datalist = np.array(
    #     [
    #         # "TaACaH",
    #         "TaAC-1H1a1H",
    #         "TaACH-1H1a1H",
    #         # "Ta1bAC-2H2b2-1aT1H",
    #     ],
    #     dtype=object,
    # )

    datalist = gandetoken(datalist[:]).tolist()
    test_set = []
    while len(test_set) < 0.15 * len(datalist):
        i = np.random.randint(0, len(datalist))
        test_set.append(datalist.pop(i))
    train_set = datalist
    print(len(test_set), len(train_set))
    print(train_set[0])
    train_set = np.asanyarray(train_set, dtype=object)
    test_set = np.asanyarray(test_set, dtype=object)
    one_hot_tensors = one_hot_encoding(train_set)
    padded_tensors = padding(one_hot_tensors)

    # Training loop
    num_epochs = 1000

    for epoch in range(num_epochs):
        model.train()
        batch_size = 64
        train_correct = 0
        train_total = 0
        for n in range(0, len(padded_tensors), batch_size):
            model.train()
            batch_input = padded_tensors[n : n + batch_size]
            batch_output = padded_tensors[n : n + batch_size]
            # Forward pass
            output = model(batch_input)

            # Calculate the loss
            loss = criterion(output, torch.argmax(batch_output, axis=1))

            train_total += batch_output.size(0)
            train_correct += (
                (output.argmax(axis=1) == batch_output.argmax(axis=1)).sum().item()
            )
            # # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # # Printing the loss
        if (epoch) % 250 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")
            print(f"Train Accuracy: {train_correct / train_total}")
            print(batch_input[0].argmax(axis=1))
            print(output[0].argmax(axis=1))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        one_hot_tensors = one_hot_encoding(test_set)
        for tensor in one_hot_tensors:
            padded_tensors = tensor.float()
            output = model(padded_tensors)
            loss = criterion(output, torch.argmax(padded_tensors, axis=1))
            test_total += padded_tensors.size(0)
            test_correct += (
                (output.argmax(axis=1) == padded_tensors.argmax(axis=1)).sum().item()
            )
    print(f"Test Accuracy: {test_correct / test_total}")
