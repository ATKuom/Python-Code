import numpy as np
import config
import pandas as pd
from split_functions import string_to_equipment, token_to_string

SQP_results = np.load(
    config.DATA_DIRECTORY / "v22DFSQP_m2_sorted_results.npy", allow_pickle=True
)
SQP_layouts = np.load(
    config.DATA_DIRECTORY / "v22DFSQP_m2_sorted_layouts.npy", allow_pickle=True
)

PSO_results = np.load(config.DATA_DIRECTORY / "v22DF_m2_results.npy", allow_pickle=True)
PSO_layouts = np.load(config.DATA_DIRECTORY / "v22DF_m2_layouts.npy", allow_pickle=True)


def dataset_prep(sorted_results, sorted_layouts):
    sorted_layouts = sorted_layouts
    sorted_results = sorted_results

    sorted_layouts = string_to_equipment(sorted_layouts)
    rearranged = []
    for equipment in sorted_layouts:
        equipment = equipment[1:-1]
        T = equipment.index(1)
        equipment = np.roll(equipment, -T).tolist()
        rearranged.append(equipment)

    rearranged_strings = token_to_string(rearranged)

    df = pd.DataFrame(rearranged_strings, columns=["Equipment"])
    df["Score"] = sorted_results
    return df


df1 = dataset_prep(SQP_results, SQP_layouts)
df2 = dataset_prep(PSO_results, PSO_layouts)
print(len(df1), len(df2))
df1.insert(2, "PSO_Results", np.nan)
for i in range(len(df1)):
    layout = df1["Equipment"].iloc[i]
    if layout in df2["Equipment"].values:
        pso_result = df2[df2["Equipment"] == layout]["Score"].values[0]
        df1["PSO_Results"].iloc[i] = pso_result

# df1.to_excel(config.DATA_DIRECTORY / "v22DF_SQP_vs_PSO.xlsx")
