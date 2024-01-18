import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import config
import torch.utils.data as data
from LSTM_batch_pack import LSTMtry, model, classes


def generation(N, model):
    generated_layouts = np.zeros(N, dtype=object)
    i = 0
    while i in range(N):
        prediction = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ).reshape(1, -1, 12)
        model.eval()
        with torch.no_grad():
            while not torch.equal(
                prediction[0][-1],
                torch.tensor(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                ),
            ):
                new_character = model(prediction)
                index = torch.multinomial(
                    nn.functional.softmax(new_character, dim=1), 1
                ).item()
                new_tensor = torch.tensor([0.0] * len(classes))
                new_tensor[index] = 1.0
                prediction = torch.cat(
                    (prediction[0], new_tensor.reshape(1, 12))
                ).reshape(1, -1, 12)
        # maxlength enforcement
        generated_layouts[i] = prediction
        i += 1
    layout_list = []
    for layout in generated_layouts:
        seq = ""
        for n in range(layout.shape[1]):
            index = layout[0, n, :].argmax(axis=0).item()
            seq += classes[index]
        layout_list.append(seq)
    layout_list = np.array(layout_list, dtype=object)
    return layout_list


if __name__ == "__main__":
    model.load_state_dict(torch.load(config.MODEL_DIRECTORY / "v8D0_m1.pt"))
    layout_list = generation(N=10000, model=model)
    np.save(config.DATA_DIRECTORY / "v810k.npy", layout_list)
