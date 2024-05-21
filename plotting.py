import matplotlib.pyplot as plt
import numpy as np

# bobj = [
#     # 133.48,
#     133.83,
#     133.45,
#     129.46,
#     129.41,
#     128.15,
#     128.11,
#     129.93,
#     129.45,
# ]
# bobj_labels = [
#     # "TAaACH-1a1T1H",
#     "TaAC-1CH1TaH1",
#     "TaAC-1C1H1aH",
#     "TaAC-1a1HT1CH",
#     "TaAC-1HT1aH1H",
#     "TaAC-1H1TaH1H",
#     "TaACT-1H1aH1H",
#     "TaAC-1HT1CaH1H",
#     "TaAC-1H1a1CTH",
# ]
# x = np.arange(len(bobj))
# regres = [-0.6377, 132.46]
# trendline = regres[0] * x + regres[1]
# plt.plot(x, trendline, label="Trendline", color="blue", linestyle="dotted")
# plt.scatter(x, bobj, label="Best Objective Function Value", marker="o", color="blue")
# for i, txt in enumerate(bobj_labels):
#     plt.annotate(
#         txt, (x[i], bobj[i]), textcoords="offset points", xytext=(0, 10), ha="center"
#     )
# plt.tick_params(direction="in")
# plt.xlabel("Iteration Steps")
# plt.ylabel("Objective Function Value ($/MWh)")
# plt.title("Best Perfoming Design Over Iteration Steps")
# plt.legend()
# plt.show()

# HPGvG = [
#     5.4,
#     15.9,
#     23.0,
#     25.2,
#     24.0,
#     25.8,
#     26.5,
#     26.2,
# ]
# HPGvN = [
#     6.1,
#     20.6,
#     34.7,
#     40.7,
#     41.6,
#     46.1,
#     48.0,
#     49.6,
# ]
# x = np.arange(len(HPGvG))
# plt.plot(x, HPGvN, label="Generated(Novel)", marker="v", color="pink")
# plt.plot(x, HPGvG, label="Generated(Total)", marker="x", color="purple")
# plt.tick_params(direction="in")
# plt.xlabel("Iteration Steps")
# plt.ylabel("Percentage of High-Performance Designs (%)")
# plt.title("M2 Generation Performance")
# plt.legend()
# plt.show()

# obj144 = [
#     0.4,
#     10.8,
#     19.9,
#     18.7,
#     19.8,
#     20.0,
#     19.8,
#     20.1,
#     20.3,
# ]
# obj164 = [
#     2.9,
#     23.7,
#     29.7,
#     30.1,
#     34.0,
#     32.9,
#     31.4,
#     32.2,
#     31.1,
# ]
# x = np.arange(len(obj144))
# plt.plot(x, obj164, label="164 $/MWh", marker="v", color="pink")
# plt.plot(x, obj144, label="144 $/MWh", marker="x", color="purple")
# plt.tick_params(direction="in")
# plt.xlabel("Iteration Steps")
# plt.ylabel("Percentage of Novel Designs (%)")
# plt.title("Obj. Fun. Value Threshold Effects on Generation Performance")
# plt.legend()
# plt.show()
# obj144 = [
#     43,
#     367,
#     963,
#     1523,
#     2116,
#     2716,
#     3311,
#     3913,
#     4522,
# ]
# obj164 = [287, 998, 1890, 2794, 3814, 4801, 5742, 6709, 7643]
# x = np.arange(len(obj144))
# plt.plot(x, obj164, label="164 $/MWh", marker="v", color="pink")
# plt.plot(x, obj144, label="144 $/MWh", marker="x", color="purple")
# plt.tick_params(direction="in")
# plt.xlabel("Iteration Steps")
# plt.ylabel("Number of High-perfomance Designs")
# plt.title("Obj. Fun. Value Threshold Effects on Dataset Size")
# plt.legend()
# plt.show()

# ds100 = [
#     100,
#     131,
#     164,
#     230,
#     380,
#     730,
#     2758,
#     7625,
#     13684,
#     20188,
#     26516,
#     33233,
#     40088,
#     46710,
#     53329,
# ]

# ds1k = [
#     1000,
#     2176,
#     7096,
#     15337,
#     24007,
#     32858,
#     41926,
#     51085,
# ]

# ds10k = [
#     10000,
#     18536,
#     27370,
#     36511,
#     45656,
#     54947,
# ]
# ds50k = [50000]

# x = np.arange(len(ds100))

# plt.plot(x, ds100, label="100", marker="x")
# plt.plot(x[: len(ds1k)], ds1k, label="1k", marker="v")
# plt.plot(x[: len(ds10k)], ds10k, label="10k", marker="s")
# plt.plot(x[: len(ds50k)], ds50k, label="50k", marker="o")
# plt.tick_params(direction="in")
# plt.xlabel("Iteration Steps")
# plt.ylabel("Dataset Size")
# plt.title("Dataset Growth Over Iteration Steps")
# plt.legend()
# plt.show()

# ds100n = [
#     31,
#     33,
#     66,
#     150,
#     350,
#     2028,
#     4867,
#     6059,
#     6504,
#     6328,
#     6717,
#     6855,
#     6622,
#     6619,
#     6813,
# ]
# ds1kn = [
#     1246,
#     5096,
#     8609,
#     9125,
#     9345,
#     9502,
#     9590,
#     9643,
# ]
# ds10kn = [
#     8536,
#     8834,
#     9141,
#     9145,
#     9291,
#     9343,
# ]

# ds50kn = [9211]

# ds100n = np.asarray(ds100n) / 100
# ds1kn = np.asarray(ds1kn) / 100
# ds10kn = np.asarray(ds10kn) / 100
# ds50kn = np.asarray(ds50kn) / 100

# x = np.arange(len(ds100n))
# plt.plot(x, ds100n, label="100", marker="x")
# plt.plot(x[: len(ds1kn)], ds1kn, label="1k", marker="v")
# plt.plot(x[: len(ds10kn)], ds10kn, label="10k", marker="s")
# plt.plot(x[: len(ds50kn)], ds50kn, label="50k", marker="o")
# plt.xlabel("Iteration Steps")
# plt.tick_params(direction="in")
# plt.ylabel("Percentage of Novel Designs (%)")
# plt.title("Generation Novelty of Models")
# plt.legend()
# plt.show()
