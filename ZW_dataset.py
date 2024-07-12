import torch
from torch.utils.data import Dataset, DataLoader
from split_functions import string_to_equipment

# from config import DATA_DIRECTORY
import numpy as np

# from ZW_utils import std_classes


class LSTMDataset(Dataset):
    def __init__(self, data, classes, training_type="standard"):
        self.base = data
        print("Designs in the dataset:", len(self.base))
        self.data = data
        self.classes = classes
        # Perform one-hot encoding for the sequences
        self.data = self.one_hot_encoding(self.data, self.classes)
        if training_type == "augmented":
            self.data = self.augment_data(self.data)
            print("Data augmented:", len(self.data) - len(self.base))
        # input output preparation
        self.data, self.labels, self.lengths = self.input_output_prep(self.data)
        # Padding
        self.data = torch.nn.utils.rnn.pad_sequence(
            self.data, batch_first=True, padding_value=0
        ).float()
        # Output classes
        self.labels = torch.argmax(self.labels, dim=1)
        print("Input shape:", self.data.shape)

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

    def augment_data(self, data):
        augmented = []
        for i in data:
            base = i.numpy()
            nognoe = base[1:-1]
            for j in range(1, len(nognoe)):
                new_rep = torch.from_numpy(np.roll(nognoe, j, axis=0))
                augmented.append(torch.cat((i[:1], new_rep, i[-1:]), axis=0))
        return data + augmented


class GPTDataset(Dataset):
    def __init__(self, data, classes, block_size, training_type="standard"):
        self.base = data
        print("Designs in the dataset:", len(self.base))
        self.data = data
        self.classes = classes
        # Integer encoding
        self.data = string_to_equipment(self.data, self.classes)
        if training_type == "augmented":
            self.data = self.augment_data(self.data)
            print("Data augmented:", len(self.data) - len(self.base))
        self.data = [i + [11] * (block_size - len(i)) for i in self.data]
        # input output preparation
        self.data, self.labels = self.input_output_prep(self.data)
        # Output classes
        print("Input shape:", self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def input_output_prep(self, data):
        inputs = torch.tensor(data[:-1], dtype=torch.long)
        outputs = torch.tensor(data[1:], dtype=torch.long)
        return inputs, outputs

    def augment_data(self, data):
        augmented = []
        for i in data:
            base = np.array(i)
            nognoe = base[1:-1]
            for j in range(1, len(nognoe)):
                new_rep = np.roll(nognoe, j, axis=0)
                augmented.append(
                    np.concatenate((base[0:1], new_rep, base[-1:]), axis=0).tolist()
                )
        return data + augmented


# datapath = DATA_DIRECTORY / "v21D10_m1.npy"
# data = np.load(datapath, allow_pickle=True)
# dataset = LSTMDataset(data, std_classes, training_type="augmented")
