import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
import config

# datalist = np.load(config.DATA_DIRECTORY / "D0test.npy", allow_pickle=True)
# example = max(datalist, key=len)
# print(example, len(example))

# ex2 = max(datalist, key=len)
# print(ex2, len(ex2))
# np.save(config.DATA_DIRECTORY / "D0test.npy", allow_pickle=True)

# classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]
# # datalist = np.array(
# #     [

# #             "GTaACaHE",
# #             "GTaAC-1H1a1HE",
# #             "GTaACH-1H1a1HE",
# #             "GTa1bAC-2H2b2-1aT1HE",

# #     ]
# # )


# def one_hot_encoding(datalist):
#     one_hot_tensors = []
#     for sequence in datalist:
#         # Perform one-hot encoding for the sequence
#         one_hot_encoded = []
#         i = 0
#         while i < len(sequence):
#             char = sequence[i]
#             vector = [0] * len(classes)  # Initialize with zeros

#             if char == "-":
#                 next_char = sequence[i + 1]
#                 unit = char + next_char
#                 if unit in classes:
#                     vector[classes.index(unit)] = 1
#                     i += 1  # Skip the next character since it forms a unit
#             elif char in classes:
#                 vector[classes.index(char)] = 1

#             one_hot_encoded.append(vector)
#             i += 1

#         # Convert the list to a PyTorch tensor
#         one_hot_tensor = torch.tensor(one_hot_encoded)
#         one_hot_tensors.append(one_hot_tensor)

#     return one_hot_tensors


# def padding(one_hot_tensors):
#     # Pad the one-hot tensors to have the same length
#     padded_tensors = pad_sequence(
#         one_hot_tensors, batch_first=True, padding_value=0
#     ).float()

#     return padded_tensors


# one_hot_tensors = one_hot_encoding(datalist)
# padded_tensors = padding(one_hot_tensors)

# print(np.load("D0test.npy", allow_pickle=True))


print(np.log(0.1))
