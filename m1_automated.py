from LSTM_generation import generation
from thermo_validity import validity
import numpy as np
import config
import torch
import torch.optim as optim
from LSTM_batch_pack import (
    LSTMtry,
    training,
    classes,
    criterion,
)

if __name__ == "__main__":
    N = 10000
    datasets = [
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10",
        # "D11",
        # "D12",
        # "D13",
        # "D14",
    ]
    next_datasets = [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10",
        "D11",
        # "D12",
        # "D13",
        # "D14",
        # "D15",
    ]
    version = "v21"
    model_phase = "_m1"

    for dataset, next_dataset in zip(datasets, next_datasets):
        datalist_name = version + dataset + model_phase + ".npy"
        model_name = version + dataset + model_phase + ".pt"
        generated_name = version + dataset + model_phase + "_generated.npy"
        next_datalist_name = version + next_dataset + model_phase + ".npy"
        datalist = np.load(
            config.DATA_DIRECTORY / datalist_name, allow_pickle=True
        ).tolist()
        model = LSTMtry(
            input_size=len(classes), hidden_size=32, num_classes=len(classes)
        )
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001,
        )
        last_model, best_model, train_acc, train_loss, val_acc, val_loss = training(
            model, optimizer, criterion, datalist, 30, 100
        )
        torch.save(best_model, config.MODEL_DIRECTORY / model_name)

        # ML Generation
        model.load_state_dict(torch.load(config.MODEL_DIRECTORY / model_name))
        layout_list = generation(N, model)
        np.save(config.DATA_DIRECTORY / generated_name, layout_list)

        # Validity Filter
        datalist = np.load(config.DATA_DIRECTORY / generated_name, allow_pickle=True)
        print(generated_name, len(datalist))
        print("V", len(validity(datalist)))
        valid_strings = np.unique(np.array(validity(datalist), dtype=object))
        print("V/U", len(valid_strings))
        p_datalist = np.load(config.DATA_DIRECTORY / datalist_name, allow_pickle=True)
        n_datalist = np.concatenate((p_datalist, valid_strings), axis=0)
        n_valid_strings = np.unique(n_datalist)
        print(next_datalist_name, len(n_valid_strings))
        np.save(config.DATA_DIRECTORY / next_datalist_name, n_valid_strings)
