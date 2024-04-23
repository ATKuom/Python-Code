"""
Vogel GPT2
input size 512 (max sequence length), 8000 PFDs
12 decoder layers
h=12 attention heads
a feedforward sublayer
embeding size = 768\
85.9M parameters
"""

##Hirtretier T5
# original small t5 60M parameters
# embedding size 128
# 2 encoder layers
# 2 decoder layers
# 7.9M parameters
##Balhorn
# T5
# 128 embedding, 4 encoder layers, 4 decoder layers, 5e-4 learning rate, 32 batch size
# Evaluation at every 20 steps, early topping patience 40
# 7.9M parameters
"""
--------------------- Questions ---------------------
In the translation examples they were putting, source and target sentences together?
How are we going to use it to generate just from the start not like a translation?
The translation spec is used in balhorn as he was trying the input of PFD and the output of the PID as the source and target translation

"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import config
from LSTM_batch_pack import (
    count_parameters,
    classes,
    training,
    optimizer,
    criterion,
    one_hot_encoding,
    padding,
)


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=12,
            num_encoder_layers=0,
            num_decoder_layers=12,
            dim_feedforward=768,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        out_x = self.embedding(x)
        out = self.transformer(out_x)
        out = self.fc(out)
        # out = self.softmax(out)
        return out


model = Transformer(input_size=len(classes), hidden_size=768, num_classes=len(classes))

if __name__ == "__main__":
    datalist = np.load(
        config.DATA_DIRECTORY / "v4D0_m1.npy", allow_pickle=True
    ).tolist()
    validation_set = []
    while len(validation_set) < 0.15 * len(datalist):
        i = np.random.randint(0, len(datalist))
        validation_set.append(datalist.pop(i))
    validation_set = np.asanyarray(validation_set, dtype=object)
    datalist = np.asanyarray(datalist, dtype=object)

    train_input, train_output = one_hot_encoding(datalist)
    padded_train_input = padding(train_input)
    # sequence_lengths = torch.tensor([x for x in map(len, train_input)])
    train_output = torch.stack(train_output)

    validation_input, validation_output = one_hot_encoding(validation_set)
    padded_validation_input = padding(validation_input)
    # validation_lengths = [x for x in map(len, validation_input)]
    validation_output = torch.stack(validation_output)
    print(datalist.shape[0], validation_set.shape[0], padded_train_input.shape[0])
    # Training loop
    best_model = None
    best_loss = np.inf
    indices = np.arange(len(padded_train_input))
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    num_epochs = 30
    batch_size = 100
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        epoch_loss = 0
        steps = 0
        model.train()

        for n in range(0, len(padded_train_input), batch_size):
            optimizer.zero_grad()
            batch_input = padded_train_input[n : n + batch_size]
            batch_output = train_output[n : n + batch_size]

            # batch_lengths = sequence_lengths[n : n + batch_size]
            # packed_batch_input = nn.utils.rnn.pack_padded_sequence(
            #     batch_input, batch_lengths, batch_first=True, enforce_sorted=False
            # )
            output = model(
                batch_input.long(),
                batch_output.view(batch_size, 1, len(classes)).long(),
            )
            loss = criterion(output, batch_output.argmax(axis=1))
            train_total += batch_output.size(0)
            train_correct += (
                (output.argmax(axis=1) == batch_output.argmax(axis=1)).sum().item()
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        epoch_loss = epoch_loss / steps
        train_loss.append(epoch_loss)
        train_acc.append(100 * train_correct / train_total)
        # np.random.shuffle(indices)
        # padded_train_input = padded_train_input[indices]
        # train_output = train_output[indices]
        # sequence_lengths = sequence_lengths[indices]

        model.eval()
        loss = 0
        correct = 0
        total = 0
        steps = 0
        with torch.no_grad():
            for n in range(0, len(padded_validation_input), batch_size):
                batch_input = padded_validation_input[n : n + batch_size]
                batch_output = validation_output[n : n + batch_size]
                # batch_lengths = validation_lengths[n : n + batch_size]

                # packed_batch_input = nn.utils.rnn.pack_padded_sequence(
                #     batch_input,
                #     batch_lengths,
                #     batch_first=True,
                #     enforce_sorted=False,
                # )
                # Forward pass
                output = model(batch_input.float())

                # Calculate the loss
                loss += criterion(output, batch_output.argmax(axis=1))
                total += batch_output.size(0)
                correct += (
                    (output.argmax(axis=1) == batch_output.argmax(axis=1)).sum().item()
                )
                steps += 1
            loss = loss / steps
            val_loss.append(loss)
            val_acc.append(100 * correct / total)
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(
                    "Epoch %d: TrainingLoss: %.3f ValidationLoss: %.3f"
                    % (epoch, epoch_loss, loss)
                )
                print(
                    "Accuracy of the network on the training set: %d %%"
                    % (100 * train_correct / train_total)
                )
                print(
                    "Accuracy of the network on the validation set: %d %%"
                    % (100 * correct / total)
                )
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    # plt.show()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.show()
