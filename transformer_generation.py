import torch
import numpy as np
import config
from TRANSFORMER_randomstrings import classes, device, model


model.load_state_dict((torch.load(config.MODEL_DIRECTORY / "TD0_m2.pt")))


context = torch.zeros((1, 1), dtype=torch.long, device=device)
int_to_char = dict((i, c) for i, c in enumerate(classes))
decode = lambda l: "".join([int_to_char[i] for i in l])
N = 3000
generated_layouts = np.zeros(N, dtype=object)
for i in range(N):
    transformer_output = model.generate(context, max_new_tokens=24)[0].tolist()
    generated_layouts[i] = decode(transformer_output)
    # if i % 500 == 0:
    #     print(i, "Generated")
print(generated_layouts)

np.save(config.DATA_DIRECTORY / "generated_layouts.npy", generated_layouts)
