import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import config
import torch.utils.data as data

classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]


def gandetoken(datalist):
    for index in range(len(datalist)):
        datalist[index] = "G" + datalist[index] + "E"
    return datalist


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


class LSTMtry(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMtry, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2
        )
        self.dlayer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, hout = self.lstm(x)
        # out1, input_sizes = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # i = 0
        # out3 = torch.zeros(out1.shape[0], 1, out1.shape[2])
        # for index in input_sizes:
        #     out3[i] = out1[i, index - 1, :]
        #     i += 1
        # out3 = out3[:, -1, :]
        out = self.dlayer(hout[0][-1])

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
    datalist = np.load(
        config.DATA_DIRECTORY / "valid_sequences.npy", allow_pickle=True
    ).tolist()
    # datalist = gandetoken(datalist[:]).tolist()

    test_set = []
    while len(test_set) < 0.15 * len(datalist):
        i = np.random.randint(0, len(datalist))
        test_set.append(datalist.pop(i))
    test_set = np.asanyarray(test_set, dtype=object)
    datalist = np.asanyarray(datalist, dtype=object)

    train_input, train_output = one_hot_encoding(datalist)
    padded_train_input = padding(train_input)
    sequence_lengths = [x for x in map(len, train_input)]
    train_output = torch.stack(train_output)

    validation_input, validation_output = one_hot_encoding(test_set)
    padded_validation_input = padding(validation_input)
    validation_lengths = [x for x in map(len, validation_input)]
    validation_output = torch.stack(validation_output)
    print(padded_train_input.shape[0], train_output.shape[0])
    # Training loop
    num_epochs = 1000
    best_model = None
    best_loss = np.inf
    acc = []
    batch = [1, 4]
    for b in batch:
        for epoch in range(num_epochs):
            batch_size = b
            train_correct = 0
            train_total = 0
            for n in range(0, len(padded_train_input), batch_size):
                model.train()
                batch_input = padded_train_input[n : n + batch_size]
                batch_output = train_output[n : n + batch_size]
                batch_lengths = sequence_lengths[n : n + batch_size]
                packed_batch_input = nn.utils.rnn.pack_padded_sequence(
                    batch_input, batch_lengths, batch_first=True, enforce_sorted=False
                )

                # Forward pass
                output = model(packed_batch_input.float())

                # output
                # Calculate the loss
                loss = criterion(
                    output, batch_output.argmax(axis=1)
                )  # torch.argmax(padded_tensors, axis=1))
                train_total += batch_output.size(0)
                train_correct += (
                    (output.argmax(axis=1) == batch_output.argmax(axis=1)).sum().item()
                )
                # # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            loss = 0
            correct = 0
            total = 0
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
                        (output.argmax(axis=1) == batch_output.argmax(axis=1))
                        .sum()
                        .item()
                    )
                if loss < best_loss:
                    best_loss = loss
                    best_model = model.state_dict()
                    train_acc = 100 * train_correct / train_total
                    val_acc = 100 * correct / total
                # if epoch % 10 == 0 or epoch == num_epochs - 1:
                # print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))
                # print(
                #     "Accuracy of the network on the training set: %d %%"
                #     % (100 * train_correct / train_total)
                # )
                # print(
                #     "Accuracy of the network on the validation set: %d %%"
                #     % (100 * correct / total)
                # )

        acc.append([train_acc, val_acc])
    print(acc)
    # model.load_state_dict(best_model)
    # prediction = torch.tensor(
    #     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # ).reshape(1, -1, 12)
    # model.eval()
    # with torch.no_grad():
    #     while not torch.equal(
    #         prediction[-1],
    #         torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    #     ):
    #         breakpoint()
    #         new_character = model(prediction)
    #         new_tensor = torch.tensor([0.0] * len(classes))
    #         new_tensor[torch.argmax(new_character).item()] = 1.0
    #         prediction = torch.cat((prediction[0], new_tensor.reshape(1, 12))).reshape(
    #             1, -1, 12
    #         )
    #         print(prediction)
