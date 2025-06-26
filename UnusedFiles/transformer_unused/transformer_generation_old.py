import torch
import torch.nn as nn
import numpy as np
import config
from trans_trial2 import model, max_seq_length, classes
from split_functions import token_to_string


def generation(N, model):
    model.eval()
    generated_layouts = np.zeros(N, dtype=object)
    for i in range(N):
        generated_seq = torch.tensor([[1]], dtype=torch.int64)
        input_sequence = torch.tensor([[]], dtype=torch.int64)
        with torch.no_grad():
            for _ in range(max_seq_length - 1):
                # Get source mask
                pred = model(input_sequence, generated_seq)
                # next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
                next_item = torch.multinomial(
                    nn.functional.softmax(pred, dim=-1)[:, -1, :], 1
                )

                # Concatenate previous input with predicted best word
                generated_seq = torch.cat((generated_seq, next_item), dim=1)

                # Stop if model predicts end of sentence
                if next_item.view(-1).item() == 12:
                    break
        generated_layouts[i] = generated_seq

    layout_list = []
    for layout in generated_layouts:
        seq = ""
        for n in range(layout.shape[1]):
            index = layout[0, n].item()
            seq += classes[index]
        layout_list.append(seq)
    layout_list = np.array(layout_list, dtype=object)
    return layout_list


if __name__ == "__main__":
    model.load_state_dict(torch.load(config.MODEL_DIRECTORY / "transformer_trial.pt"))
    layout_list = generation(N=100, model=model)
    print(layout_list)
    # np.save(config.DATA_DIRECTORY / "TTv1.npy", layout_list)
