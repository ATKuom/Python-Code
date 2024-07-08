from pathlib import Path
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import os
from time import strftime
import copy

# Standart class
std_classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]

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
def data_loaders(data, batch_size, data_split_ratio):
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


# Loss functions
std_loss = nn.CrossEntropyLoss()


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
        print(
            "Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f%%"
            % (epoch + 1, max_epochs, epoch_loss, 100 * train_correct / train_total)
        )
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (x, y, lengths) in enumerate(val_loader):
                packed_x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                y_pred = model(packed_x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            print(
                "Accuracy of the network on the validation set: %d %%"
                % (100 * correct / total)
            )
    return model


def make_dir(model, batch_size, learning_rate):
    time = strftime("%Y%m%d%H%M")
    learning_rate_sci = "{:.0e}".format(learning_rate)
    model_name = model.__class__.__name__
    directory = f"{time}_{model_name}_batch{batch_size}_lr{learning_rate_sci}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
