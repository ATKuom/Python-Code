import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import config
import torch.utils.data as data
import time
import matplotlib.pyplot as plt

classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]
s = time.time()


def one_hot_encoding(datalist):
    one_hot_tensors = []
    train_input = []
    train_output = []
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
        # one_hot_tensor = torch.tensor(one_hot_encoded)
        # one_hot_tensors.append(one_hot_tensor)
        for i in range(len(one_hot_encoded) - 1):
            train_input.append(torch.tensor(one_hot_encoded[0 : i + 1]))
            train_output.append(torch.tensor(one_hot_encoded[i + 1]))
    return train_input, train_output


def padding(one_hot_tensors):
    # Pad the one-hot tensors to have the same length
    padded_tensors = pad_sequence(
        one_hot_tensors, batch_first=True, padding_value=0
    ).float()
    return padded_tensors  # .view(-1, len(classes))


def training(model, optimizer, criterion, datalist, num_epochs=30, batch_size=32):
    datalist_length = len(datalist)
    print(
        "datalist_length:",
        datalist_length,
        "Total_epoch:",
        num_epochs,
        "Batch_size:",
        batch_size,
    )
    validation_set = []

    while len(validation_set) < 0.15 * datalist_length:
        i = np.random.randint(0, len(datalist))
        validation_set.append(datalist.pop(i))
    validation_set = np.asanyarray(validation_set, dtype=object)
    datalist = np.asanyarray(datalist, dtype=object)

    train_input, train_output = one_hot_encoding(datalist)
    padded_train_input = padding(train_input)
    sequence_lengths = torch.tensor([x for x in map(len, train_input)])
    train_output = torch.stack(train_output)

    validation_input, validation_output = one_hot_encoding(validation_set)
    padded_validation_input = padding(validation_input)
    validation_lengths = [x for x in map(len, validation_input)]
    validation_output = torch.stack(validation_output)
    print(datalist.shape[0], validation_set.shape[0], padded_train_input.shape[0])
    best_model = None
    best_loss = np.inf
    indices = np.arange(len(padded_train_input))
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        epoch_loss = 0
        steps = 0
        model.train()

        for n in range(0, len(padded_train_input), batch_size):
            optimizer.zero_grad()
            batch_input = padded_train_input[n : n + batch_size]
            batch_output = train_output[n : n + batch_size]
            batch_lengths = sequence_lengths[n : n + batch_size]
            packed_batch_input = nn.utils.rnn.pack_padded_sequence(
                batch_input, batch_lengths, batch_first=True, enforce_sorted=False
            )
            output = model(packed_batch_input.float())
            loss = criterion(output, batch_output.argmax(axis=1))
            train_total += batch_output.size(0)
            train_correct += (
                (output.argmax(axis=1) == batch_output.argmax(axis=1)).sum().item()
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        epoch_loss = epoch_loss / steps
        train_loss.append(epoch_loss)
        train_acc.append(100 * train_correct / train_total)
        np.random.shuffle(indices)
        padded_train_input = padded_train_input[indices]
        train_output = train_output[indices]
        sequence_lengths = sequence_lengths[indices]

        model.eval()
        loss = 0
        correct = 0
        total = 0
        steps = 0
        with torch.no_grad():
            for n in range(0, len(padded_validation_input), batch_size):
                batch_input = padded_validation_input[n : n + batch_size]
                batch_output = validation_output[n : n + batch_size]
                batch_lengths = validation_lengths[n : n + batch_size]

                packed_batch_input = nn.utils.rnn.pack_padded_sequence(
                    batch_input,
                    batch_lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )
                # Forward pass
                output = model(packed_batch_input.float())

                # Calculate the loss
                loss += criterion(output, batch_output.argmax(axis=1))
                total += batch_output.size(0)
                correct += (
                    (output.argmax(axis=1) == batch_output.argmax(axis=1)).sum().item()
                )
                steps += 1
            loss = loss / steps
            val_loss.append(loss)
            val_acc.append(100 * correct / total)
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(
                    "Epoch %d: TrainingLoss: %.3f ValidationLoss: %.3f"
                    % (epoch, epoch_loss, loss)
                )
                print(
                    "Accuracy of the network on the training set: %d %%"
                    % (100 * train_correct / train_total)
                )
                print(
                    "Accuracy of the network on the validation set: %d %%"
                    % (100 * correct / total)
                )
    return best_model, train_acc, train_loss, val_acc, val_loss


class LSTMtry(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMtry, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            # dropout=0.2,
        )
        self.dlayer = nn.Linear(hidden_size, num_classes)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        # output, input_sizes = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # i = 0
        # out3 = torch.zeros(out1.shape[0], 1, out1.shape[2])
        # for index in input_sizes:
        #     out3[i] = out1[i, index - 1, :]
        #     i += 1
        # out3 = out3[:, -1, :]
        output = hidden[-1]
        out = self.dlayer(output)

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
    model.load_state_dict(torch.load(config.MODEL_DIRECTORY / "v4D10_m1.pt"))
    datalist = np.load(
        config.DATA_DIRECTORY / "v4D8_m2_candidates.npy", allow_pickle=True
    ).tolist()
    best_model, train_acc, train_loss, val_acc, val_loss = training(
        model, optimizer, criterion, datalist, 30, 32
    )
    e = time.time()
    print(e - s)
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.show()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.show()
    # torch.save(best_model, config.MODEL_DIRECTORY / "v3D0_m2.pt")
