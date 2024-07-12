from ZW_model import *
from ZW_utils import *
import torch.optim as optim
import matplotlib.pyplot as plt

dataset_id = "v21D0_m1.npy"
batch_size = 10
learning_rate = 1e-3
data_split_ratio = 0.85
augmentation = True
block_size = 22
loss_function = std_loss
max_epochs = 20
n_embd = 32  # 32
n_head = 4  # 4
n_layer = 2  # 2
dropout = 0.1  # 0.1
classes = std_classes
vocab_size = len(classes)
model = GPTModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

save_path = make_dir(
    model,
    batch_size,
    learning_rate,
)

dataset = dataloading(dataset_id)

train_loader, val_loader = T_integer_data_loaders(
    dataset, batch_size, data_split_ratio, classes, block_size, augmentation
)

best_model, last_model, t_loss, v_loss = T_integer_training(
    model, optimizer, loss_function, train_loader, val_loader, max_epochs
)
plt.plot(t_loss, label="Training Loss")
plt.plot(v_loss, label="Validation Loss")
plt.legend()
plt.savefig(f"{save_path}/loss.png")
torch.save(best_model, f"{save_path}/model.pt")
