"""
import torch

# Specify your classes
classes = ["GO", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "EOS"]

# Example designs
expert_designs = [
    "TaACaH",
    "TaAC-1H1a1H",
    "TaACH-1H1a1H",
    "Ta1bAC-2H2b2-1aT1H",
]

# Process each expert design
for expert_design in expert_designs:
    print("Expert Design:", expert_design)

    # Perform one-hot encoding for the expert design
    one_hot_encoded = []
    i = 0
    while i < len(expert_design):
        char = expert_design[i]
        vector = [0] * len(classes)  # Initialize with zeros

        if char == "-":
            if i + 1 < len(expert_design):
                next_char = expert_design[i + 1]
                unit = char + next_char
                if unit in classes:
                    vector[classes.index(unit)] = 1
                    i += 1  # Skip the next character since it forms a unit
        elif char in classes:
            vector[classes.index(char)] = 1

        one_hot_encoded.append(vector)
        i += 1

    # Convert the list to a PyTorch tensor
    one_hot_tensor = torch.tensor(one_hot_encoded)

    # Create the answer from the one-hot encoded tensor
    answer = ""
    for vector in one_hot_tensor:
        class_index = torch.argmax(vector).item()
        char = classes[class_index]
        answer += char

    print("One-Hot Encoded Output:")
    print(one_hot_tensor)
    print()
    print("Answer:")
    print(answer)
    print("-------------------------------------------------------")
    """

import torch
import numpy as np
from designs import goeos_expert


# Specify your classes
classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]

# Example designs
datalist = goeos_expert

# Process each expert design
for sequence in datalist:
    # Perform one-hot encoding for the sequence
    one_hot_encoded = []
    i = 0
    while i < len(sequence):
        char = sequence[i]
        vector = [0] * len(classes)  # Initialize with zeros

        if char == "-":
            next_char = sequence[i + 1]
            unit = char + next_char
            if unit in classes:
                vector[classes.index(unit)] = 1
                i += 1  # Skip the next character since it forms a unit
        elif char in classes:
            vector[classes.index(char)] = 1

        one_hot_encoded.append(vector)
        i += 1

    # Convert the list to a PyTorch tensor
    one_hot_tensor = torch.tensor(one_hot_encoded)

    # Create the string form from the one-hot encoded tensor
    string_form = "".join(
        classes[torch.argmax(vector).item()] for vector in one_hot_tensor
    )

    # Output the string form
    print("Sequence:", sequence)
    print("One-Hot Encoded Output:")
    print(one_hot_tensor)
    print("String Form:")
    print(string_form)
    print("-------------------------------------------------------")
