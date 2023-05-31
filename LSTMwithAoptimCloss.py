import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim


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


# Specify your classes
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

one_hot_tensors = one_hot_encoding(datalist)
padded_tensors = padding(one_hot_tensors)

# Define the LSTM model
input_size = len(classes)
hidden_size = 32
num_layers = 2
output_size = len(classes)
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

# Define the dense output layer
dense_layer = nn.Linear(hidden_size, output_size)

# Initialize the hidden state
batch_size = len(datalist)
hidden = (
    torch.zeros(num_layers, batch_size, hidden_size).float(),
    torch.zeros(num_layers, batch_size, hidden_size).float(),
)

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer
learning_rate = 0.001
optimizer = optim.Adam(
    list(lstm.parameters()) + list(dense_layer.parameters()),
    learning_rate,
)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    output, hidden = lstm(padded_tensors, hidden)

    output = dense_layer(output)

    # Calculate the loss
    loss = loss_fn(output, torch.argmax(padded_tensors, axis=1))

    # # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # # Print the loss for every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Print the final output and hidden state
