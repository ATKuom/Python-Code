import config
import numpy as np

# layouts = np.load(config.DATA_DIRECTORY / "penalty_layouts.npy", allow_pickle=True)
# print(len(layouts))
classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]
char_to_int = dict((c, i) for i, c in enumerate(classes))
int_to_char = dict((i, c) for i, c in enumerate(classes))

# sequences = []
# for layout in layouts:
#     x = ""
#     for i in range(len(layout)):
#         x += int_to_char[layout[i].argmax().item()]
#     sequences.append(x)

# print(len(sequences), len(layouts))
# sequences = np.array(sequences, dtype=object)
# np.save(config.DATA_DIRECTORY / "valid_sequences.npy", sequences)
