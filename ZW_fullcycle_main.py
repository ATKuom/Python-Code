from thermo_validity import *
from ZW_model import LSTM
from ZW_utils import *
from ZW_generation import *
import torch.optim as optim
import matplotlib.pyplot as plt

dataset_id = "v21D0_m1.npy"
classes = std_classes
data_split_ratio = 0.85
batch_size = 100
max_epochs = 30
learning_rate = 0.001
model = LSTM()
loss_function = std_loss
augmentation = True
N = 10_000
save_path = make_dir(
    model,
    batch_size,
    learning_rate,
)
dataset = dataloading(dataset_id)

for i in range(11):
    train_loader, val_loader = data_loaders(
        dataset, batch_size, data_split_ratio, classes, augmentation
    )
    model = LSTM()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_model, last_model, t_loss, v_loss = packed_training(
        model, optimizer, loss_function, train_loader, val_loader, max_epochs
    )
    plot_name = str(i) + "_loss.png"
    model_name = str(i) + "_model.pt"
    data_name = str(i) + "_data.npy"
    plt.plot(t_loss, label="Training Loss")
    plt.plot(v_loss, label="Validation Loss")
    plt.legend()
    plt.savefig(f"{save_path}/{plot_name}")
    plt.clf()
    torch.save(best_model, f"{save_path}/{model_name}")
    model.load_state_dict(best_model)
    layout_list = generation(N, model=model)
    np.save(f"{save_path}/generated+{data_name}", layout_list)
    new_strings = np.array(validity(layout_list), dtype=object)
    prev_strings = np.array(dataset, dtype=object)
    new_dataset = np.unique(np.concatenate((prev_strings, new_strings), axis=0))
    np.save(f"{save_path}/generated+{data_name}", new_dataset)
    dataset = new_dataset.tolist()
