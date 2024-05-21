import config
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

##Combining all good layouts into one file
np.set_printoptions(threshold=np.inf)


def dataset_combination(version):
    a = [
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        # "D9",
    ]
    b1 = version
    b2 = "_results.npy"
    b3 = "_candidates.npy"
    model = "_m2"
    good_layouts = []
    good_results = []
    for a in a:
        result = b1 + a + model + b2
        candidate = b1 + a + model + b3
        results = np.load(config.DATA_DIRECTORY / result, allow_pickle=True)
        datalist = np.load(config.DATA_DIRECTORY / candidate, allow_pickle=True)
        nonzero_results = results[np.where(results > 0)]
        cutoff = 143.957  # 164.428
        print(len(nonzero_results), len(results), len(datalist))
        for i in range(len(results)):
            if results[i] < cutoff and results[i] > 0:
                good_layouts.append(datalist[i])
                good_results.append(results[i])
        print(len(good_layouts), len(good_results))
    good_layouts = np.array(good_layouts, dtype=object)
    good_results = np.array(good_results, dtype=object)
    final_dataset_name = b1 + model + "_layouts.npy"
    final_results_name = b1 + model + b2
    np.save(config.DATA_DIRECTORY / final_dataset_name, good_layouts)
    np.save(config.DATA_DIRECTORY / final_results_name, good_results)
    return


# Finding the good layouts
datalist = np.load(config.DATA_DIRECTORY / "TD1_m2_candidates.npy", allow_pickle=True)
results = np.load(config.DATA_DIRECTORY / "TD1_m2_results2310.npy", allow_pickle=True)
positions = np.load(
    config.DATA_DIRECTORY / "TD1_m2_positions2310.npy", allow_pickle=True
)
nonzero_results = results[np.where(results > 0)]
cutoff = 143.957
# cutoff = 30000
good_layouts = []
good_results = []
good_positions = []
best_result = cutoff
print(len(nonzero_results), len(results), len(datalist))
for i in range(len(results)):
    if results[i] < cutoff and results[i] > 0:
        good_layouts.append(datalist[i])
        good_results.append(results[i])
        good_positions.append(positions[i])
        if results[i] < best_result:
            best_result = results[i]
            best_layout = datalist[i]
print(len(good_layouts), len(good_results), len(good_positions))
print(best_result, best_layout)
print(np.mean(good_results))
good_layouts = np.array(good_layouts, dtype=object)
good_results = np.array(good_results, dtype=object)
good_positions = np.array(good_positions, dtype=object)
np.save(config.DATA_DIRECTORY / "TD1g_m2.npy", good_layouts)
np.save(config.DATA_DIRECTORY / "TD1g_m2_results.npy", good_results)
np.save(config.DATA_DIRECTORY / "TD1g_m2_positions.npy", good_positions)

quit()
# layouts = good_layouts
# results = good_results

##Final good layouts graphical analysis
layouts = np.load(config.DATA_DIRECTORY / "v26D8_m2_candidates.npy", allow_pickle=True)
results = np.load(
    config.DATA_DIRECTORY / "v26D8_m2_results.npy",
    allow_pickle=True,
)
print(len(layouts), len(results))


layouts2 = []
results2 = []
good_results = []
good_layouts = []
cutoff = 130.87 * 1.1
for i in range(len(results)):
    if results[i] > 0:
        results2.append(results[i])
        layouts2.append(layouts[i])
        if results[i] < cutoff:
            good_results.append(results[i])
            good_layouts.append(layouts[i])
layouts = np.asanyarray(layouts2)
results = np.asanyarray(results2)

print(len(layouts), len(results))
# std = np.std(results)
# mean = np.mean(results)
# print(
#     std,
#     mean,
#     mean - std * 3,
#     mean - std * 2,
#     mean - std,
#     mean + std,
#     mean + std * 2,
#     mean + std * 3,
# )
# n, bins, patches = plt.hist(
#     results,
#     bins=[
#         0,
#         mean - std * 3,
#         mean - std * 2,
#         mean - std,
#         mean + std,
#         mean + std * 2,
#         mean + std * 3,
#     ],
# )
# print(n)


ed1 = 134.69
ed2 = 130.87
ed3 = 134.52
bins = [ed2 * x / 100 for x in range(80, 111)]
lessthan_ed2 = np.where(results < ed2)[0]
lessthan_ed1 = np.where(results < ed1)[0]
lessthan_ed3 = np.where(results < ed3)[0]
group1 = len(lessthan_ed2)
group2 = len(lessthan_ed3) - group1
group3 = len(lessthan_ed1) - group2 - group1
n, bins, patches = plt.hist(results, bins=bins, color="green")
print(sum(n))
xlabel = "Exergo-economic cost ($/MWh)"
ylabel = "Frequency"
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(" Top-p(0.75) Sampling")
plt.axvline(ed2, color="red", linestyle="dashed", linewidth=1, label="ED2")
plt.axvline(ed1, color="black", linestyle="dashed", linewidth=1, label="ED1")
plt.axvline(ed3, color="blue", linestyle="dashed", linewidth=1, label="ED3")
plt.text(110, 45, "              c<130.87: " + str(group1))
plt.text(110, 25, "130.87<c<134.52: " + str(group2))
plt.text(110, 5, "134.52<c<134.69: " + str(group3))
# plt.xlim(120, 150)
# plt.ylim(0, 25)
plt.legend()
# plt.show()
indices = np.argsort(results)
sorted_results = results[indices]
sorted_layouts = layouts[indices]
print(
    "mean of good layouts",
    np.mean(good_results),
    "mean of all layouts",
    np.mean(sorted_results),
)
# print(sorted_layouts[: int(sum(n))], sorted_results[: int(sum(n))])
print(sorted_layouts[:10], sorted_results[:10])

# np.save(
#     config.DATA_DIRECTORY / "v4DF_sorted_results.npy",
#     sorted_results,
# )
# np.save(
#     config.DATA_DIRECTORY / "v23D0_m2.npy",
#     good_layouts,
# )

##Positional information gathering
# positions = np.load(config.DATA_DIRECTORY / "v21D8_m2_positions.npy", allow_pickle=True)
# for layout in sorted_layouts:
#     i = np.where(layouts == layout)[0][0]
#     print(layout, results[i], positions[i])
#     break

##Detailed analysis of ED_Test results
# analysis = np.load(
#     config.DATA_DIRECTORY / "len20m2v2_final_sorted_layouts_lessthanED1_results.npy",
#     allow_pickle=True,
# )
# parameter = []
# for x in analysis[:]:
#     parameter.append(x[5])
# parameter = np.asarray(parameter).reshape(-1, 1)
# print(parameter)

##Creation of the first dataset based on diferent approaches
# layouts = np.load(
#     config.DATA_DIRECTORY / "v3D0.npy",
#     allow_pickle=True,
# )
# results = np.load(
#     config.DATA_DIRECTORY / "D0_results.npy",
#     allow_pickle=True,
# )
# index = np.load(
#     config.DATA_DIRECTORY / "D0_valid.npy",
#     allow_pickle=True,
# )
# i = 0
# j = 0
# valid_layouts = layouts[index]
# print(len(layouts), len(valid_layouts))
# new_dataset = valid_layouts
# for layout in new_dataset:
#     if "1" in layout:
#         j += 1
#         if "a" in layout:
#             i += 1
# print(i, j)
# total_i = i
# i = total_i
# j = 0
# indices = np.arange(len(layouts))
# np.random.shuffle(indices)
# layouts = layouts[indices].tolist()
# for layout in valid_layouts:
#     layouts.remove(layout)
# print(len(layouts), len(new_dataset))
# new_dataset = new_dataset.tolist()
# for layout in layouts:
#     if "1" in layout:
#         j += 1
#         if "a" in layout:
#             new_dataset.append(layout)
#             layouts.remove(layout)
#     if len(new_dataset) == 737:
#         break
# for layout in layouts:
#     new_dataset.append(layout)
#     if len(new_dataset) == 1000:
#         break
# j = 0
# i = 0
# k = 0
# for layout in new_dataset:
#     if "a" in layout:
#         k += 1
#     if "1" in layout:
#         j += 1
#         if "a" in layout:
#             i += 1
# print(len(new_dataset), i, j, k)
# np.save(config.DATA_DIRECTORY / "v3D0_m1.npy", new_dataset)

# results1 = np.load(
#     config.DATA_DIRECTORY / "v4D0_5k_results.npy",
#     allow_pickle=True,
# )
# results2 = np.load(
#     config.DATA_DIRECTORY / "v4D0_5k+_results.npy",
#     allow_pickle=True,
# )
# for i in range(len(results1)):
#     if results1[i] > results2[i]:
#         results2[i] = results1[i]
# np.save(config.DATA_DIRECTORY / "v4D0_10k_results.npy", results2)
