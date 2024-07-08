import torch
from torch.utils.data import Dataset, DataLoader
from config import DATA_DIRECTORY
import numpy as np


class LSTMDataset(Dataset):
    def __init__(self, data, classes):
        self.data = data
        self.classes = classes
        # Perform one-hot encoding for the sequences
        self.data = self.one_hot_encoding(self.data, self.classes)
        # input output preparation
        self.data, self.labels, self.lengths = self.input_output_prep(self.data)
        # Padding
        self.data = torch.nn.utils.rnn.pad_sequence(
            self.data, batch_first=True, padding_value=0
        ).float()
        # Output classes
        self.labels = torch.argmax(self.labels, dim=1)
        print("Input shape:", self.data.shape)
        print("Output shape:", self.labels.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.lengths[idx]

    def one_hot_encoding(self, data, classes):
        one_hot_tensors = []
        for sequence in data:
            one_hot_encoded = []
            i = 0
            while i < len(sequence):
                char = sequence[i]
                vector = [0] * len(classes)
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

    def input_output_prep(self, one_hot_tensors):
        input = []
        output = []
        for one_hot_encoded in one_hot_tensors:
            for i in range(len(one_hot_encoded) - 1):
                input.append(one_hot_encoded[0 : i + 1])
                output.append(one_hot_encoded[i + 1])
        lengths = [len(i) for i in input]
        output = torch.stack(output)
        lengths = torch.tensor(lengths)
        return input, output, lengths


# datapath = DATA_DIRECTORY / "v21D0_m1.npy"
# data = np.load(datapath, allow_pickle=True)
# dataset = LSTMDataset(datapath, classes)
