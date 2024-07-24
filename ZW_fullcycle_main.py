from thermo_validity import *
from ZW_model import LSTM
from ZW_utils import *
from ZW_generation import *
import torch.optim as optim
import matplotlib.pyplot as plt
from ZW_Opt import *
from split_functions import one_hot_encoding, bound_creation

dataset_id = "v21D0_m1.npy"
classes = std_classes
data_split_ratio = 0.85
batch_size = 100
max_epochs = 30
learning_rate = 0.001
model = LSTM()
loss_function = std_loss
augmentation = False
N = 10_000
cutoff = 143.957

# save_path = make_dir(
#     model,
#     batch_size,
#     learning_rate,
# )
# dataset = dataloading(dataset_id)

save_path = "202407151014_LSTM_NA"
dataset = np.load(f"{save_path}/generated+7_data.npy", allow_pickle=True).tolist()


def optimization(data_array, classes, save_path, save_name):
    one_hot_tensors = np.array(one_hot_encoding(data_array, classes), dtype=object)
    print(one_hot_tensors.shape)
    valid_layouts = set()
    penalty_layouts = set()
    broken_layouts = set()
    results = np.zeros(one_hot_tensors.shape[0])
    positions = np.zeros(one_hot_tensors.shape[0], dtype=object)
    for i in range(one_hot_tensors.shape[0]):
        layout = one_hot_tensors[i]
        equipment, bounds, x, splitter = bound_creation(layout)
        # PSO Parameters
        swarmsize_factor = 7
        particle_size = swarmsize_factor * len(bounds)
        if 5 in equipment:
            particle_size += -1 * swarmsize_factor
        if 9 in equipment:
            particle_size += -2 * swarmsize_factor
        iterations = 30
        nv = len(bounds)
        try:
            a = PSO(
                objective_function, bounds, particle_size, iterations, nv, equipment
            )
            if a.result < 1e6:
                valid_layouts.add(i)
                results[i] = a.result
                positions[i] = a.points
            else:
                penalty_layouts.add(i)
        except:
            broken_layouts.add(i)
        if i % 100 == 0:
            print(
                "Valid/Penalty/Broken",
                len(valid_layouts),
                len(penalty_layouts),
                len(broken_layouts),
            )
    results_name = "results_" + save_name + ".npy"
    positions_name = "positions_" + save_name + ".npy"
    np.save(f"{save_path}\{results_name}", results)
    np.save(f"{save_path}\{positions_name}", positions)
    return results


def optimization_filter(results, datalist, cutoff, save_name):
    nonzero_results = results[np.where(results > 0)]
    good_layouts = []
    good_results = []
    print("Optimization Results:", len(nonzero_results), len(results))
    for i in range(len(results)):
        if results[i] < cutoff and results[i] > 0:
            good_layouts.append(datalist[i])
            good_results.append(results[i])
    print("Good layouts", len(good_layouts))
    good_layouts = np.array(good_layouts, dtype=object)
    np.save(
        f"{save_path}/{save_name}_good_layouts.npy",
        good_layouts,
    )
    return good_layouts, good_results


for i in range(8, 11):
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

model.load_state_dict(torch.load(f"{save_path}/10_model.pt"))
initial_10k = generation(N, model=model)
initial_10k = np.unique(np.array(validity(initial_10k), dtype=object))
savefile_name = "initial_10k"
np.save(f"{save_path}/{savefile_name}.npy", initial_10k)
# initial_10k = np.load(f"{save_path}/initial_10k.npy", allow_pickle=True)
print(len(initial_10k))
results = optimization(initial_10k, classes, save_path, savefile_name)
# results = np.load(f"{save_path}/results_initial_10k.npy")
initial_good_layouts, initial_good_results = optimization_filter(
    results, initial_10k, cutoff, savefile_name
)
print(np.sort(np.array(initial_good_results), axis=0))
