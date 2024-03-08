import torch
import numpy as np
import config
from trans_trial4 import classes, device, model


model.load_state_dict(
    (torch.load(config.MODEL_DIRECTORY / "transformer_trial_lastmodel.pt"))
)


context = torch.zeros((1, 1), dtype=torch.long, device=device)
int_to_char = dict((i, c) for i, c in enumerate(classes))
decode = lambda l: "".join([int_to_char[i] for i in l])
N = 10000
generated_layouts = np.zeros(N, dtype=object)
for i in range(N):
    transformer_output = model.generate(context, max_new_tokens=25)[0].tolist()
    generated_layouts[i] = decode(transformer_output)
    if i % 500 == 0:
        print(i, "Generated")
print(generated_layouts)

np.save(config.DATA_DIRECTORY / "generated_layouts.npy", generated_layouts)
