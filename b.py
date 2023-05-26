import torch
import numpy as np
import torch.nn.functional as F
import time

start_time = time.time()

# Specify your classes
classes = ["T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2"]

# Example designs
datalist = np.array(["TaACaH", "TaAC-1H1a1H", "TaACH-1H1a1H", "Ta1bAC-2H2b2-1aT1H"])

# Initialize lists to store one-hot tensors and generated strings
one_hot_tensors = []
generated_strings = []

# Process each expert design
for sequence in datalist:
    # Perform one-hot encoding for the sequence
    indices = []
    i = 0
    while i < len(sequence):
        char = sequence[i]
        if char == "-":
            if i + 1 < len(sequence):
                unit = sequence[i : i + 2]
                if unit in classes:
                    indices.append(classes.index(unit))
                    i += 1  # Skip the next character since it forms a unit
        else:
            indices.append(classes.index(char))
        i += 1

    one_hot_tensor = F.one_hot(torch.tensor(indices), num_classes=len(classes)).float()

    # Create the string form from the one-hot encoded tensor
    string_form = "".join(
        classes[torch.argmax(vector).item()] for vector in one_hot_tensor
    )

    # Append the one-hot tensor and generated string to the lists
    one_hot_tensors.append(one_hot_tensor)
    generated_strings.append(string_form)

end_time = time.time()
print("Total time:", end_time - start_time)

# Compare inputs and outputs
for i, sequence in enumerate(datalist):
    print("Input:", sequence)
    print("One-Hot Tensor:")
    print(one_hot_tensors[i])
    print("Generated String:", generated_strings[i])
    print("-------------------------------------------------------")
