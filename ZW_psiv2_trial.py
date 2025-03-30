from ZW_utils import std_classes, dataloading
from ZW_dataset import PSI_Dataset
import numpy as np
from config import DATA_DIRECTORY
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from split_functions import uniqueness_check
from ZW_model import GPT
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


# layouts = np.load(DATA_DIRECTORY/"v22DF_m2_sorted_layouts.npy", allow_pickle=True)
# results = np.load(DATA_DIRECTORY/"v22DF_m2_sorted_results.npy", allow_pickle=True)
# layouts = np.load("GPT_NA/initial_10k.npy", allow_pickle=True)
# results = np.load("GPT_NA/results_initial_10k.npy", allow_pickle=True)
# l2 = []
# r2 = []
# cutoff = 500
# for i, r in enumerate(results):
#     if r > 0 and r < cutoff:
#         l2.append(layouts[i])
#         r2.append(r)
# layouts = np.asanyarray(l2)
# results = np.asanyarray(r2)


# designs, equipments = uniqueness_check(layouts)
# sorted_equipments = equipments.copy()
# sorted_equipments.sort()
# sorted_results = []
# for se in sorted_equipments:
#     index = equipments.index(se)
#     sorted_results.append(results[index])
# eq_array = np.zeros((len(sorted_equipments), 22))
# for i, e in enumerate(sorted_equipments):
#     for j, u in enumerate(e):
#         eq_array[i, j] = u
# re_array = np.array(sorted_results)
# equipment_chunks = []
# results_chunks = []
# for equipment in sorted_equipments:
#     for i in range(len(equipment)):
#         candidate_chunk = equipment[: i + 1]
#         if candidate_chunk not in equipment_chunks:
#             equipment_chunks.append(candidate_chunk)
#             # checking the same chunks in eq array
#             chunk_indices = np.where(
#                 (eq_array[:, : i + 1] == candidate_chunk).all(axis=1)
#             )[0]
#             chunk_results = np.mean(re_array[chunk_indices])
#             results_chunks.append(chunk_results)
# print(25,equipment_chunks[25], results_chunks[25])

# lengths = torch.tensor([x for x in map(len, equipment_chunks)])
# max_length = max(lengths)
# input_data = np.ones((len(equipment_chunks), max_length)) * 12
# for i, e in enumerate(equipment_chunks):
#     input_data[i, : len(e)] = e
# input_data = torch.tensor(input_data)
# target_data = torch.tensor(results_chunks).float().reshape(-1, 1)
# print(input_data.shape, target_data.shape)

# # normalizing the target data to be between 0 and 1
# # print(target_data.min().item(), target_data.max().item())
# target_data = (target_data - target_data.min()) / (target_data.max() - target_data.min())
# # standardizing the target data
# # target_data = (target_data - target_data.mean()) / target_data.std()

# indices = torch.randperm(len(input_data))
# input_data = input_data[indices]
# target_data = target_data[indices]
# lengths = lengths[indices]
# train_data = input_data[: int(0.85 * len(input_data))]
# train_target = target_data[: int(0.85 * len(input_data))]
# train_lengths = lengths[: int(0.85 * len(input_data))]
# val_data = input_data[int(0.85 * len(input_data)) :]
# val_target = target_data[int(0.85 * len(input_data)) :]
# val_lengths = lengths[int(0.85 * len(input_data)) :]
# print(train_data[25], train_target[25], train_lengths[25])

# for embd_size in [64]:
#     for hidden_size in [1024]:
#         batch_size = 8
#         patience = 10
#         model = LSTM_packed(embd_size,hidden_size)
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#         best_loss = 1e9
#         for epoch in range(max_epochs+1):
#             model.train()
#             epoch_loss = 0
#             for i in range(0, len(train_data), batch_size):
#                 input_batch = train_data[i : i + batch_size]
#                 target_batch = train_target[i : i + batch_size]
#                 lengths_batch = train_lengths[i : i + batch_size]
#                 optimizer.zero_grad()
#                 output = model(input_batch, lengths_batch)
#                 loss = criterion(output, target_batch)
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item()
#             epoch_loss /= len(train_data) / batch_size

#             indices = torch.randperm(len(train_data))
#             train_data = train_data[indices]
#             train_target = train_target[indices]
#             train_lengths = train_lengths[indices]

#             model.eval()
#             val_loss = 0
#             with torch.no_grad():
#                 for i in range(0, len(val_data), batch_size):
#                     input_batch = val_data[i : i + batch_size]
#                     target_batch = val_target[i : i + batch_size]
#                     lengths_batch = val_lengths[i : i + batch_size]
#                     output = model(input_batch, lengths_batch)
#                     loss = criterion(output, target_batch)
#                     val_loss += loss.item()
#                 val_loss /= len(val_data) / batch_size
#             if val_loss < best_loss:
#                 best_model_epoch = epoch
#                 best_loss = val_loss
#                 best_model = model.state_dict()
#                 patience = 10
#             else:
#                 patience -= 1
#             if patience == 0:
#                 break
#             # print(f"Epoch {epoch} Training Loss: {epoch_loss:.2f} Validation Loss: {val_loss:.2f}")
#             #random prediction
#             random_index = np.random.randint(0, len(val_data))
#             random_input = val_data[random_index]
#             random_target = val_target[random_index]
#             random_length = val_lengths[random_index]
#             random_output = model(random_input.unsqueeze(0), random_length.unsqueeze(0))
#             # print(f"Target: {random_target.item():.2f} Prediction: {random_output.item():.2f} Error: {abs(random_target.item() - random_output.item())/random_target.item()*100:.2f}")
#         torch.save(best_model, f"psi_std_{embd_size}_{hidden_size}_{batch_size}.pt")
#         # best model prediction and mean error
#         model.load_state_dict(best_model)
#         model.eval()
#         print("Best Model Prediction",embd_size,hidden_size)
#         print("batch_size",batch_size,"found epoch",best_model_epoch)
#         mean_error = 0
#         with torch.no_grad():
#             for i in range(0, len(val_data)):
#                 input_batch = val_data[i].unsqueeze(0)
#                 target_batch = val_target[i].unsqueeze(0)
#                 lengths_batch = val_lengths[i].unsqueeze(0)
#                 output = model(input_batch, lengths_batch)
#                 if target_batch.item() == 0:
#                     continue
#                 mean_error += ((torch.abs(output - target_batch))/torch.abs(target_batch)).item()
#             mean_error /= len(val_data)
#             print(f"Mean Error: {mean_error*100:.2f}%")
#             for i in range(5):
#                 random_index = np.random.randint(0, len(val_data))
#                 random_input = val_data[random_index]
#                 random_target = val_target[random_index]
#                 random_length = val_lengths[random_index]
#                 random_output = model(random_input.unsqueeze(0), random_length.unsqueeze(0))
#                 print(f"Target: {random_target.item():.2f} Prediction: {random_output.item():.2f} Error: {abs((random_target.item() - random_output.item())/random_target.item())*100:.2f}")


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
phi = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)

# predicor
psi = LSTM_packed(64, 256)


phi.load_state_dict(torch.load("GPT_NA_psitest/M2_model_8.pt"))
psi.load_state_dict(torch.load("PSI_models/psi_norm_min_aug_100max64_256_4_300_8.pt"))

phi.eval()
psi.eval()
N = 3000
int_to_char = dict((i, c) for i, c in enumerate(classes))
decode = lambda l: "".join([int_to_char[i] for i in l])
equipment_list = []
string_list = []
for i in range(N):
    psi_token_stack = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).reshape(
        12, 1
    )
    idx = torch.zeros((1, 1), dtype=torch.long)
    for _ in range(22):
        phi_logits = phi(idx)
        phi_logits = phi_logits[:, -1, :]
        idx_stack = idx.repeat(12, 1)
        idx_stack = torch.cat((idx_stack, psi_token_stack), 1).float()
        lengths = torch.tensor([idx_stack.size(1)] * idx_stack.size(0))
        print(idx_stack, idx_stack.shape)
        print(lengths, lengths.shape)
        quit()
        psi_logits = psi(idx_stack, lengths)
        # product = (phi_logits.flatten() + 1 * (1 - psi_logits.flatten())).reshape(1, 12)
        # product2 = torch.mul(phi_logits.flatten(), (1 - psi_logits.flatten()))
        phi_prob = F.softmax(phi_logits.flatten(), dim=-1)
        product3 = torch.mul(phi_prob, (100 - psi_logits.flatten()))
        # probs = F.softmax(product, dim=-1)
        # secondway
        # probs = F.softmax(product2, dim=-1).reshape(1, 12)
        # thirdway
        probs = (product3 / sum(product3)).reshape(1, 12)
        # phi_probs = F.softmax(phi_logits.flatten(), dim=-1)
        # probs = torch.mul(phi_probs, psi_logits.flatten())
        k = 1
        topp = probs.topk(k)
        total_prob = topp[0].sum()
        while total_prob < 0.9:
            k += 1
            topp = probs.topk(k)
            total_prob = topp[0].sum()
        idx_next = topp[1][0][torch.multinomial(topp[0] / total_prob, 1)]
        idx = torch.cat((idx, idx_next), dim=1)
        if idx_next.item() == len(classes) - 1:
            break
    idx = idx.flatten().tolist()
    string_list.append(decode(idx))
    equipment_list.append(idx)

from thermo_validity import validity

cutoff = 143.957
save_path = "GPT_NA_psitest"
dataset = np.load("GPT_NA_psitest/M2_data_8.npy", allow_pickle=True)
generated_layouts = string_list
print("Number of generated layouts: ", len(generated_layouts))
print("Number of valid layouts: ", len(validity(generated_layouts)))
print("Number of unique valid layouts: ", len(np.unique(validity(generated_layouts))))
unique_strings = np.unique(np.array(validity(generated_layouts), dtype=object))
# p_datalist = dataset
# datalist = np.unique(np.concatenate((p_datalist, unique_strings), axis=0))
# # Separating the new strings from the old ones
# candidates = datalist[np.where(np.isin(datalist, p_datalist, invert=True))[0]]
# print("Number of unique valid new layouts: ", len(candidates))
np.save(f"{save_path}/psiphi_generated_M2_0_min_aug_300_100max.npy", generated_layouts)
# for e,s in zip(equipment_list,string_list):
#     print(e,s)
# for e,s in zip(equipment_list,string_list):
#     print(e,s)
# string_list = np.load(
#     "GPT_NA_psitest/psiphi_generated_M2_0_3rdway_144.npy", allow_pickle=True
# )
from ZW_Transmain import optimization, optimization_filter

candidates = unique_strings
i = 0
# Separating the new strings from the old ones
# print("previous data length:", len(p_datalist))
print("candidates length:", len(candidates))
# Optimization of the new strings
candidates_results = optimization(
    candidates, classes, save_path, "candidates_0_psi_min_aug_300_100max" + str(i)
)
# Filtering the results above the threshold
good_layouts, good_results = optimization_filter(
    candidates_results,
    candidates,
    cutoff,
    save_path,
    "M2_0_psi_min_aug_300_100max" + str(i),
)
