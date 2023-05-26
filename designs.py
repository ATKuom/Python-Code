import numpy as np

expert_designs = [
    "TaACaH",
    "TaAC-1H1a1H",
    "TaACH-1H1a1H",
    "Ta1bAC-2H2b2-1aT1H",
    # "Ta1bAC-2H2b2-3H3a-1T13H",
]
arr_expert = np.array(expert_designs)
goeos_expert = []
for sequence in expert_designs:
    sequence = "G" + sequence + "E"
    goeos_expert.append(sequence)
print(goeos_expert)

goeos_expert = ["GTaACaHE", "GTaAC-1H1a1HE", "GTaACH-1H1a1HE", "GTa1bAC-2H2b2-1aT1HE"]
