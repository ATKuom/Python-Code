import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import config
import torch.utils.data as data

classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]


def gandetoken(datalist):
    """
    Start (G) and stop (E) tokens are added to the power plant layouts
    """
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
    padded_tensors = pad_sequence(
        one_hot_tensors, batch_first=True, padding_value=0
    ).float()
    return padded_tensors  # .view(-1, len(classes))


class LSTMtry(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # self.lstm = nn.LSTM(
        #     input_size,
        #     hidden_size,
        #     num_layers=2,
        #     batch_first=True,
        #     # dropout=0.2,
        # )
        self.l1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.l2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dlayer = nn.Linear(hidden_size, num_classes)
        self.bn = nn.BatchNorm1d(
            hidden_size,
            momentum=0.01,
        )

    def forward(self, x):
        # out, hlstm = self.lstm(x)
        # out = self.bn(hlstm[0][-1])
        o1, (h1, c1) = self.l1(x)
        o2, (h2, c2) = self.l2(o1)
        out = h2[-1]
        # out = hlstm[0][-1]
        out = self.dlayer(out)

        return out


model = LSTMtry(input_size=len(classes), hidden_size=32, num_classes=len(classes))

criterion = nn.CrossEntropyLoss()

# Define the optimizer
learning_rate = 0.001
optimizer = optim.Adam(
    model.parameters(),
    learning_rate,
    weight_decay=0.0001,
)
if __name__ == "__main__":
    datalist = np.load(config.DATA_DIRECTORY / "D0.npy", allow_pickle=True)

    datalist = gandetoken(datalist[:]).tolist()
    validation_set = []
    while len(validation_set) < 0.15 * len(datalist):
        i = np.random.randint(0, len(datalist))
        validation_set.append(datalist.pop(i))

    validation_set = np.asanyarray(validation_set, dtype=object)
    datalist = np.asanyarray(datalist, dtype=object)

    train_input, train_output = one_hot_encoding(datalist)
    validation_input, validation_output = one_hot_encoding(validation_set)

    padded_train_input = padding(train_input)
    train_output = torch.stack(train_output)
    padded_validation_input = padding(validation_input)
    validation_output = torch.stack(validation_output)
    print(datalist.shape[0], validation_set.shape[0])

    # Training loop
    num_epochs = 60
    best_model = None
    best_loss = np.inf
    train_accuracy = []
    validation_accuracy = []
    train_loss = []
    test_loss = []
    indices = np.arange(len(padded_train_input))
    for epoch in range(num_epochs):
        batch_size = 32
        train_correct = 0
        train_total = 0
        t_loss = 0
        steps = 0
        for n in range(0, len(padded_train_input), batch_size):
            model.train()
            batch_input = padded_train_input[n : n + batch_size]
            batch_output = train_output[n : n + batch_size]

            output = model(batch_input)
            loss = criterion(output, batch_output.argmax(axis=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * len(batch_output)

            train_total += batch_output.size(0)
            train_correct += (
                (output.argmax(axis=1) == batch_output.argmax(axis=1)).sum().item()
            )
            steps += 1
        t_loss = t_loss / steps
        train_loss.append(t_loss)
        np.random.shuffle(indices)
        padded_train_input = padded_train_input[indices]
        train_output = train_output[indices]
        # Validation Step
        model.eval()
        v_loss = 0
        v_correct = 0
        v_total = 0
        v_steps = 0
        with torch.no_grad():
            batch_size = 1
            for n in range(0, len(padded_validation_input), batch_size):
                batch_input = padded_validation_input[n : n + batch_size]
                batch_output = validation_output[n : n + batch_size]
                # Forward pass
                output = model(batch_input)

                # Calculate the loss
                v_loss += criterion(output, batch_output.argmax(axis=1))
                v_total += batch_output.size(0)
                v_correct += (
                    (output.argmax(axis=1) == batch_output.argmax(axis=1)).sum().item()
                )
                v_steps += 1
            v_loss = v_loss / v_steps
            test_loss.append(v_loss)
            if v_loss < best_loss:
                best_loss = v_loss
                best_model = model.state_dict()
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print("Epoch %d: TrainingLoss: %.4f" % (epoch, t_loss))
                print("Epoch %d: ValidationLoss: %.4f" % (epoch, v_loss))
                print(
                    "Accuracy of the network on the training set: %d %%"
                    % (100 * train_correct / train_total)
                )
                print(
                    "Accuracy of the network on the validation set: %d %%"
                    % (100 * v_correct / v_total)
                )

            train_accuracy.append(100 * train_correct / train_total)
            validation_accuracy.append(100 * v_correct / v_total)
    import matplotlib.pyplot as plt

    plt.plot(train_accuracy, label="Training Accuracy")
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.legend()
    plt.show()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(test_loss, label="Validation Loss")
    plt.legend()
    plt.show()

    torch.save(best_model, config.MODEL_DIRECTORY / "LSTM_batch.pt")
