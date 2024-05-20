import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
import torch.utils.data as data
from LSTM_batch_pack import LSTMtry, model, classes


def sampling(softmax_output):
    ## Sampling
    sampling_index = torch.multinomial(softmax_output, 1).item()
    return sampling_index


def greedy_search(softmax_output):
    ##Greedy search
    greedy_search_index = softmax_output.topk(1)[1].item()
    return greedy_search_index


def topp_sampling(softmax_output):
    ## top p sampling
    k = 1
    topp = softmax_output.topk(k)
    total_prob = topp[0].sum()
    while total_prob < 0.5:
        k += 1
        topp = softmax_output.topk(k)
        total_prob = topp[0].sum()
    topp_sampling_index = topp[1][0][
        torch.multinomial(topp[0] / total_prob, 1).item()
    ].item()
    return topp_sampling_index


def topk_sampling(softmax_output):
    ## top k sampling
    k = 3
    topkk = softmax_output.topk(k)
    topk_sampling_index = topkk[1][0][torch.multinomial(topkk[0], 1).item()].item()
    return topk_sampling_index


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
                softmax_output = F.softmax(model(prediction), dim=1)
                new_tensor = torch.tensor([0.0] * len(classes))
                new_tensor[topp_sampling(softmax_output)] = 1.0
                prediction = torch.cat(
                    (prediction[0], new_tensor.reshape(1, 12))
                ).reshape(1, -1, 12)
        # maxlength enforcement is not implemented. Do we need it? I am not sure.
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
    model.load_state_dict(torch.load(config.MODEL_DIRECTORY / "v22D0_m1.pt"))
    layout_list = generation(N=10000, model=model)
    print(layout_list)
    np.save(config.DATA_DIRECTORY / "v26D0_m2_candidates.npy", layout_list)
