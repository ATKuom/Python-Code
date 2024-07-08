from ZW_dataset import LSTMDataset
from ZW_model import LSTM
from ZW_utils import *
import torch.optim as optim
import matplotlib.pyplot as plt


dataset_id = "v21D3_m1.npy"
classes = std_classes
data_split_ratio = 0.85
batch_size = 100
max_epochs = 30
learning_rate = 0.001


dataset = dataloading(dataset_id)

data = LSTMDataset(dataset, classes)
train_loader, val_loader = data_loaders(data, batch_size, data_split_ratio)

model = LSTM()
loss_function = std_loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# save_path = make_dir(model, batch_size, learning_rate)


best_model, last_model, t_loss, v_loss = std_training(
    model, optimizer, loss_function, train_loader, val_loader, max_epochs
)
# plt.plot(t_loss, label="Training Loss")
# plt.plot(v_loss, label="Validation Loss")
# plt.legend()
# torch.save(M1, f"{save_path}/model.pt")
