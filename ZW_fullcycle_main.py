from thermo_validity import *
from ZW_model import LSTM
from ZW_utils import *
from ZW_generation import *
import torch.optim as optim
import matplotlib.pyplot as plt
from ZW_Opt import *
from split_functions import (
    one_hot_encoding,
    bound_creation,
    uniqueness_check,
)

dataset_id = "D0_500.npy"
classes = std_classes
data_split_ratio = 0.85
batch_size = 100
max_epochs = 30
learning_rate = 0.001
model = LSTM()
loss_function = std_loss
augmentation = "val_aug"
uniqueness = False
N1 = 10_000
cycles1 = 11
N2 = 3_000
cycles2 = 8
cutoff = 143.957


save_path = "LSTM_FA"
# save_path = make_dir(
#     model,
#     batch_size,
#     learning_rate,
# )

# dataset = dataloading(dataset_id)
dataset = np.load(f"{save_path}/M1_data_8.npy", allow_pickle=True).tolist()
if uniqueness:
    dataset, _ = uniqueness_check(dataset)


# do not forget to change the inside
def LSTM_training_cycle(mode, N, save_path, dataset, cycles, starting_cycle=0):
    if mode == "M1":
        for i in range(starting_cycle, cycles):
            train_loader, val_loader = data_loaders(
                dataset, batch_size, data_split_ratio, classes, augmentation
            )
            model = LSTM()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            best_model, last_model, t_loss, v_loss = packed_training(
                model, optimizer, loss_function, train_loader, val_loader, max_epochs
            )
            plot_name = mode + "_loss_" + str(i) + ".png"
            model_name = mode + "_model_" + str(i) + ".pt"
            data_name = mode + "_data_" + str(i + 1) + ".npy"
            plt.plot(t_loss, label="Training Loss")
            plt.plot(v_loss, label="Validation Loss")
            plt.legend()
            plt.savefig(f"{save_path}/{plot_name}")
            plt.clf()
            torch.save(best_model, f"{save_path}/{model_name}")
            model.load_state_dict(best_model)
            layout_list = generation(N, model=model)
            if uniqueness:
                layout_list, _ = uniqueness_check(layout_list)
            np.save(f"{save_path}/generated_{mode}_{i}.npy", layout_list)
            new_strings = np.array(validity(layout_list), dtype=object)
            prev_strings = np.array(dataset, dtype=object)
            new_dataset = np.unique(np.concatenate((prev_strings, new_strings), axis=0))
            np.save(f"{save_path}/{data_name}", new_dataset)
            dataset = new_dataset.tolist()
            print("Dataset Length:", len(dataset))
    else:
        for i in range(starting_cycle, cycles):
            train_loader, val_loader = data_loaders(
                dataset, batch_size, data_split_ratio, classes, augmentation
            )
            model = LSTM()
            model.load_state_dict(torch.load(f"{save_path}/M1_model_10.pt"))
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            best_model, last_model, t_loss, v_loss = packed_training(
                model,
                optimizer,
                loss_function,
                train_loader,
                val_loader,
                max_epochs,
            )
            plot_name = mode + "_loss_" + str(i) + ".png"
            model_name = mode + "_model_" + str(i) + ".pt"
            data_name = mode + "_data_" + str(i + 1) + ".npy"
            plt.plot(t_loss, label="Training Loss")
            plt.plot(v_loss, label="Validation Loss")
            plt.legend()
            plt.savefig(f"{save_path}/{plot_name}")
            plt.clf()
            torch.save(best_model, f"{save_path}/{model_name}")
            model.load_state_dict(best_model)
            # Generation from new model
            generated_layouts = generation(N, model=model)
            if uniqueness:
                generated_layouts, _ = uniqueness_check(generated_layouts)
            np.save(f"{save_path}/generated_{mode}_{i}.npy", generated_layouts)
            unique_strings = np.unique(
                np.array(validity(generated_layouts), dtype=object)
            )
            p_datalist = dataset
            datalist = np.unique(np.concatenate((p_datalist, unique_strings), axis=0))
            # Separating the new strings from the old ones
            candidates = datalist[
                np.where(np.isin(datalist, p_datalist, invert=True))[0]
            ]
            # Optimization of the new strings
            candidates_results = optimization(
                candidates, classes, save_path, "candidates_" + str(i)
            )
            # Filtering the results above the threshold
            good_layouts, good_results = optimization_filter(
                candidates_results, candidates, cutoff, "M2_" + str(i)
            )
            print(np.sort(np.array(good_results), axis=0)[:10])
            # Saving the good layouts of new and old as the new dataset
            dataset = np.unique(np.concatenate((p_datalist, good_layouts), axis=0))
            np.save(f"{save_path}/{data_name}", dataset)
    return model


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


if __name__ == "__main__":
    # M1_model = LSTM_training_cycle(
    #     "M1", N1, save_path, dataset, cycles1, starting_cycle=8
    # )
    # model.load_state_dict(torch.load(f"{save_path}/M1_model_10.pt"))
    # initial_10k = generation(10_000, model=model)
    # if uniqueness:
    #     initial_10k, _ = uniqueness_check(initial_10k)
    # initial_10k = np.unique(np.array(validity(initial_10k), dtype=object))
    # savefile_name = "initial_10k"
    # print(len(initial_10k))
    # np.save(f"{save_path}/{savefile_name}.npy", initial_10k)
    # results = optimization(initial_10k, classes, save_path, savefile_name)
    # # initial_10k = np.load(f"{save_path}/initial_10k.npy", allow_pickle=True)
    # # results = np.load(f"{save_path}/results_initial_10k.npy")
    # initial_good_layouts, initial_good_results = optimization_filter(
    #     results, initial_10k, cutoff, savefile_name
    # )
    # print(np.sort(np.array(initial_good_results), axis=0))
    initial_good_layouts = np.load(f"{save_path}/M2_data_6.npy", allow_pickle=True)
    M2_model = LSTM_training_cycle(
        "M2", N2, save_path, initial_good_layouts, cycles2, starting_cycle=6
    )
