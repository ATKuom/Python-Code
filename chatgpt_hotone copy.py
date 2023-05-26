import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# Specify your classes
classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]

# Example designs
datalist = ["GTaACaHE", "GTaAC-1H1a1HE", "GTaACH-1H1a1HE", "GTa1bAC-2H2b2-1aT1HE"]

# Initialize a list to store the one-hot tensors
one_hot_tensors = []

# Compute the maximum sequence length
max_sequence_length = max(len(seq) for seq in datalist)

# Process each expert design
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

# Pad the one-hot tensors to have the same length
padded_tensors = pad_sequence(one_hot_tensors, batch_first=True, padding_value=0)

# Print the LSTM input
print(padded_tensors)
