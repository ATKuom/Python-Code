from pathlib import Path
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import os
from time import strftime
import copy
from ZW_dataset import *

# Standart class
std_classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]
# Loss functions
std_loss = nn.CrossEntropyLoss()

# Directories
DATA_DIRECTORY = Path.home() / "Python" / "data"
MODEL_DIRECTORY = Path.home() / "Python" / "models"


# Dataloading
def dataloading(dataset_id):
    data_path = DATA_DIRECTORY / dataset_id
    dataset = np.load(data_path, allow_pickle=True).tolist()
    print("Designs in the dataset", dataset_id, ":", len(dataset))
    return dataset


# Data split and loaders
def data_loaders(dataset, batch_size, data_split_ratio, classes, augmentation):
    if augmentation == False:
        data = LSTMDataset(dataset, classes, training_type="standard")
        training_set, validation_set = torch.utils.data.random_split(
            data,
            [
                int(data_split_ratio * len(data)),
                len(data) - int(data_split_ratio * len(data)),
            ],
        )
    else:
        t_data = dataset[: int(data_split_ratio * len(dataset))]
        v_data = dataset[int(data_split_ratio * len(dataset)) :]
        training_set = LSTMDataset(t_data, classes, training_type="augmented")
        validation_set = LSTMDataset(v_data, classes, training_type="standard")
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    print("Training set size:", len(training_set))
    return train_loader, val_loader


def T_integer_data_loaders(
    dataset, batch_size, data_split_ratio, classes, block_size, augmentation
):
    if augmentation == False:
        print("No augmentation")
        data = GPTDataset(dataset, classes, block_size, training_type="standard")
        training_set, validation_set = torch.utils.data.random_split(
            data,
            [
                int(data_split_ratio * len(data)),
                len(data) - int(data_split_ratio * len(data)),
            ],
        )
    if augmentation == True:
        print("Training set augmentation")
        t_data = dataset[: int(data_split_ratio * len(dataset))]
        v_data = dataset[int(data_split_ratio * len(dataset)) :]
        training_set = GPTDataset(
            t_data, classes, block_size, training_type="augmented"
        )
        validation_set = GPTDataset(
            v_data, classes, block_size, training_type="standard"
        )
    if augmentation == "val_aug":
        print("Fully Augmented")
        data = GPTDataset(dataset, classes, block_size, training_type="augmented")
        training_set, validation_set = torch.utils.data.random_split(
            data,
            [
                int(data_split_ratio * len(data)),
                len(data) - int(data_split_ratio * len(data)),
            ],
        )
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    print("Training set size:", len(training_set))
    return train_loader, val_loader


def T_context_integer_data_loaders(
    dataset, batch_size, data_split_ratio, classes, block_size, augmentation
):
    if augmentation == False:
        data = contextGPTDataset(dataset, classes, block_size, training_type="standard")
        training_set, validation_set = torch.utils.data.random_split(
            data,
            [
                int(data_split_ratio * len(data)),
                len(data) - int(data_split_ratio * len(data)),
            ],
        )
    else:
        t_data = dataset[: int(data_split_ratio * len(dataset))]
        v_data = dataset[int(data_split_ratio * len(dataset)) :]
        training_set = contextGPTDataset(
            t_data, classes, block_size, training_type="augmented"
        )
        validation_set = contextGPTDataset(
            v_data, classes, block_size, training_type="standard"
        )
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    print("Training set size:", len(training_set))
    return train_loader, val_loader


# Training
def std_training(model, optimizer, loss_function, train_loader, val_loader, max_epochs):
    best_loss = np.inf
    best_model = None
    train_losses = []
    validation_losses = []
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0
        for i, (x, y, _) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        epoch_loss = epoch_loss / len(train_loader)
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for i, (x, y, _) in enumerate(val_loader):
                y_pred = model(x)
                loss = loss_function(y_pred, y)
                val_loss += loss.item()
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            val_loss = val_loss / len(val_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
            train_losses.append(epoch_loss)
            validation_losses.append(val_loss)
            print(
                "Epoch [%d/%d], T.Loss: %.4f, V.Loss: %.4f, T.Acc: %.2f%%, V.Acc: %d%%"
                % (
                    epoch + 1,
                    max_epochs,
                    epoch_loss,
                    val_loss,
                    100 * train_correct / train_total,
                    100 * correct / total,
                )
            )
    return best_model, model, train_losses, validation_losses


def packed_training(
    model, optimizer, loss_function, train_loader, val_loader, max_epochs
):
    best_loss = np.inf
    best_model = None
    train_losses = []
    validation_losses = []
    patience = 5
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0
        for i, (x, y, lengths) in enumerate(train_loader):
            optimizer.zero_grad()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            y_pred = model(packed_x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        epoch_loss = epoch_loss / len(train_loader)
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for i, (x, y, lengths) in enumerate(val_loader):
                packed_x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                y_pred = model(packed_x)
                loss = loss_function(y_pred, y)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                patience = 5
            else:
                patience -= 1
            train_losses.append(epoch_loss)
            validation_losses.append(val_loss)
            print(
                "Epoch [%d/%d], T.Loss: %.4f, V.Loss: %.4f, T.Acc: %.2f%%, V.Acc: %d%%"
                % (
                    epoch + 1,
                    max_epochs,
                    epoch_loss,
                    val_loss,
                    100 * train_correct / train_total,
                    100 * correct / total,
                )
            )
            if patience == 0:
                return best_model, model, train_losses, validation_losses
    return best_model, model, train_losses, validation_losses


def T_integer_training(
    model, optimizer, loss_function, train_loader, val_loader, max_epochs
):
    best_loss = np.inf
    best_model = None
    train_losses = []
    validation_losses = []
    patience = 5
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred = y_pred.view(-1, y_pred.size(-1))
            y = y.view(-1)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        epoch_loss = epoch_loss / len(train_loader)
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                y_pred = model(x)
                y_pred = y_pred.view(-1, y_pred.size(-1))
                y = y.view(-1)
                loss = loss_function(y_pred, y)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                patience = 5
            else:
                patience -= 1
            train_losses.append(epoch_loss)
            validation_losses.append(val_loss)
            print(
                "Epoch [%d/%d], T.Loss: %.4f, V.Loss: %.4f, T.Acc: %.2f%%, V.Acc: %d%%"
                % (
                    epoch + 1,
                    max_epochs,
                    epoch_loss,
                    val_loss,
                    100 * train_correct / train_total,
                    100 * correct / total,
                )
            )
            if patience == 0:
                return best_model, model, train_losses, validation_losses
    return best_model, model, train_losses, validation_losses


def make_dir(model, batch_size, learning_rate):
    time = strftime("%Y%m%d%H%M")
    learning_rate_sci = "{:.0e}".format(learning_rate)
    model_name = model.__class__.__name__
    directory = f"{time}_{model_name}_batch{batch_size}_lr{learning_rate_sci}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def transformer_generation(model, classes, N):
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long)
    int_to_char = dict((i, c) for i, c in enumerate(classes))
    decode = lambda l: "".join([int_to_char[i] for i in l])
    layout_list = []
    for _ in range(N):
        transformer_output = model.generate(context, 22, 22, classes)[0].tolist()
        layout_list.append(decode(transformer_output))
    return layout_list


def RL_integer_data_loaders(
    dataset, batch_size, data_split_ratio, classes, block_size, augmentation
):
    if augmentation == False:
        data = RLDataset(dataset, classes, block_size, training_type="standard")
        training_set, validation_set = torch.utils.data.random_split(
            data,
            [
                int(data_split_ratio * len(data)),
                len(data) - int(data_split_ratio * len(data)),
            ],
        )
    else:
        t_data = dataset[: int(data_split_ratio * len(dataset))]
        v_data = dataset[int(data_split_ratio * len(dataset)) :]
        training_set = RLDataset(t_data, classes, block_size, training_type="augmented")
        validation_set = RLDataset(
            v_data, classes, block_size, training_type="standard"
        )
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    print("Training set size:", len(training_set))
    return train_loader, val_loader
