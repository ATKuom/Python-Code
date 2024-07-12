from ZW_model import LSTM
from ZW_utils import *
import torch.optim as optim
import matplotlib.pyplot as plt

dataset_id = "v21D0_m1.npy"
classes = std_classes
data_split_ratio = 0.85
batch_size = 100
max_epochs = 30
learning_rate = 0.001
model = LSTM()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = std_loss
augmentation = True

save_path = make_dir(
    model,
    batch_size,
    learning_rate,
)

dataset = dataloading(dataset_id)

train_loader, val_loader = data_loaders(
    dataset, batch_size, data_split_ratio, classes, augmentation
)
best_model, last_model, t_loss, v_loss = packed_training(
    model, optimizer, loss_function, train_loader, val_loader, max_epochs
)
plt.plot(t_loss, label="Training Loss")
plt.plot(v_loss, label="Validation Loss")
plt.legend()
plt.savefig(f"{save_path}/loss.png")
torch.save(best_model, f"{save_path}/model.pt")
