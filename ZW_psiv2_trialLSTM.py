from ZW_utils import std_classes, dataloading
from ZW_dataset import PSI_Dataset
import numpy as np
from config import DATA_DIRECTORY
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from split_functions import uniqueness_check
from ZW_model import LSTM
import torch.nn.functional as F

classes = std_classes
data_split_ratio = 0.85
batch_size = 8
max_epochs = 30
learning_rate = 1e-3
block_size = 22
n_embd = 32  # 32
n_head = 4  # 4
n_layer = 2  # 2
dropout = 0.1  # 0.1
vocab_size = len(classes)
criterion = nn.MSELoss()


class LSTM_packed(nn.Module):
    def __init__(self, embd_size, hidden_size):
        super(LSTM_packed, self).__init__()
        self.embedding = nn.Embedding(13, embd_size)
        self.lstm = nn.LSTM(
            embd_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        x = self.embedding(x.long())
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        output, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x


# generator
phi = LSTM()

# predicor
psi = LSTM_packed(64, 256)


phi.load_state_dict(torch.load("LSTM_NA_psitest/M2_model_7.pt"))
psi.load_state_dict(torch.load("LSTM_psi_norm_aug_64_256_4_144_7.pt"))

phi.eval()
psi.eval()
N = 3000
psi_max = 1
int_to_char = dict((i, c) for i, c in enumerate(classes))
decode = lambda l: "".join([int_to_char[i] for i in l])
equipment_list = []
string_list = []
for i in range(N):
    psi_token_stack = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).reshape(
        12, 1
    )
    prediction = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ).reshape(1, -1, 12)
    idx = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        while not torch.equal(
            prediction[0][-1],
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ):
            phi_prob = F.softmax(phi(prediction), dim=1).reshape(12)
            idx_stack = idx.repeat(12, 1)
            idx_stack = torch.cat((idx_stack, psi_token_stack), 1).float()
            lengths = torch.tensor([idx_stack.size(1)] * idx_stack.size(0))
            psi_logits = psi(idx_stack, lengths)
            # normalize psi_logits with psi_max
            psi_logits = psi_logits / psi_logits.max() * psi_max
            product3 = torch.mul(phi_prob, (psi_max - psi_logits.flatten()))
            probs = (product3 / sum(product3)).reshape(1, 12)
            new_tensor = torch.tensor([0.0] * len(classes))
            next_token = torch.multinomial(probs, 1).item()
            new_tensor[next_token] = 1.0
            prediction = torch.cat((prediction[0], new_tensor.reshape(1, 12))).reshape(
                1, -1, 12
            )
            idx = torch.cat((idx, torch.tensor([[next_token]])), dim=1)

        idx = idx.flatten().tolist()
        string_list.append(decode(idx))
        equipment_list.append(idx)

from thermo_validity import validity

cutoff = 143.957
save_path = "LSTM_NA_psitest"
dataset = np.load("GPT_NA_psitest/M2_data_7.npy", allow_pickle=True)
generated_layouts = string_list
print("Number of generated layouts: ", len(generated_layouts))
print("Number of valid layouts: ", len(validity(generated_layouts)))
print("Number of unique valid layouts: ", len(np.unique(validity(generated_layouts))))
unique_strings = np.unique(np.array(validity(generated_layouts), dtype=object))
# # p_datalist = dataset
# # datalist = np.unique(np.concatenate((p_datalist, unique_strings), axis=0))
# # # Separating the new strings from the old ones
# # candidates = datalist[np.where(np.isin(datalist, p_datalist, invert=True))[0]]
# # print("Number of unique valid new layouts: ", len(candidates))
np.save(f"{save_path}/psiphi_generated_M2_7_aug_144.npy", generated_layouts)
# # for e,s in zip(equipment_list,string_list):
# #     print(e,s)
# # for e,s in zip(equipment_list,string_list):
# #     print(e,s)
# # string_list = np.load(
# #     "GPT_NA_psitest/psiphi_generated_M2_0_3rdway_144.npy", allow_pickle=True
# # )
from ZW_Transmain import optimization, optimization_filter

candidates = unique_strings
i = 0
# Separating the new strings from the old ones
# print("previous data length:", len(p_datalist))
print("candidates length:", len(candidates))
# Optimization of the new strings
candidates_results = optimization(
    candidates, classes, save_path, "candidates_7_psi_aug_144" + str(i)
)
# Filtering the results above the threshold
good_layouts, good_results = optimization_filter(
    candidates_results, candidates, cutoff, save_path, "M2_7_psi_aug" + str(i)
)
