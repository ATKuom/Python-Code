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
# plt.plot(x, obj164, label="Lenient Threshold (1.25)", marker="s", color="seagreen")
# plt.plot(x, obj144, label="Stricter Threshold (1.10)", marker="v", color="darkmagenta")
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
# plt.plot(x, obj164, label="Lenient Threshold (1.25)", marker="s", color="seagreen")
# plt.plot(x, obj144, label="Stricter Threshold (1.10)", marker="v", color="darkmagenta")
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

# plt.plot(x, ds100, label="100", marker="s")
# plt.plot(x[: len(ds1k)], ds1k, label="1k", marker="v")
# plt.plot(x[: len(ds10k)], ds10k, label="10k", marker="o")
# plt.plot(x[: len(ds50k)], ds50k, label="50k", marker="X")
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
# plt.plot(x, ds100n, label="100", marker="s")
# plt.plot(x[: len(ds1kn)], ds1kn, label="1k", marker="v")
# plt.plot(x[: len(ds10kn)], ds10kn, label="10k", marker="o")
# plt.plot(x[: len(ds50kn)], ds50kn, label="50k", marker="X")
# plt.xlabel("Iteration Steps")
# plt.tick_params(direction="in")
# plt.ylabel("Percentage of Novel Designs (%)")
# plt.title("Generation Novelty of Models")
# plt.legend()
# plt.show()


# ## Decoding strategies in M1
# sampling = [
#     1000,
#     2208,
#     5422,
#     9314,
#     16791,
#     25571,
#     34383,
#     43132,
#     52019,
#     61004,
#     70000,
#     79102,
# ]

# topp9 = [
#     1000,
#     2685,
#     10040,
#     18739,
#     27550,
#     36333,
#     45160,
#     53846,
#     62727,
#     71553,
#     80417,
#     89433,
# ]

# topp75 = [
#     1000,
#     1429,
#     3274,
#     5984,
#     8056,
#     10793,
#     13273,
#     15459,
#     17618,
#     19437,
#     21067,
#     22294,
# ]

# topp50 = [
#     1000,
#     1102,
#     1282,
#     1666,
#     1707,
#     1734,
#     1770,
#     1781,
#     1804,
#     1817,
#     1824,
#     1875,
# ]

# topk5 = [
#     1000,
#     3030,
#     7541,
#     15820,
#     24542,
#     33252,
#     42135,
#     51080,
#     59955,
#     68800,
#     77889,
#     86999,
# ]

# topk3 = [
#     1000,
#     1576,
#     4044,
#     8110,
#     12612,
#     17908,
#     23042,
#     27920,
#     33268,
#     38272,
#     43709,
#     49121,
# ]

# sampling = [
#     14.22,
#     34.98,
#     40.69,
#     77.84,
#     91.32,
#     92.46,
#     92.34,
#     93.42,
#     93.87,
#     94.55,
#     95.37,
# ]
# # sampling = [
# #     12.08,
# #     32.14,
# #     38.92,
# #     74.77,
# #     87.8,
# #     88.12,
# #     87.49,
# #     88.87,
# #     89.85,
# #     89.96,
# #     91.02,
# # ]

# topp9 = [
#     21,
#     83.09,
#     94.9,
#     95.97,
#     96.49,
#     97.24,
#     96.99,
#     97.74,
#     97.44,
#     97.64,
#     98.09,
# ]
# # topp9 = [
# #     16.85,
# #     73.55,
# #     86.99,
# #     88.11,
# #     87.83,
# #     88.27,
# #     86.86,
# #     88.81,
# #     88.26,
# #     88.64,
# #     90.16,
# # ]
# topp75 = [
#     11.36,
#     48.98,
#     96.43,
#     97.94,
#     99.25,
#     99.68,
#     99.73,
#     99.82,
#     99.67,
#     99.59,
#     99.6,
# ]
# # topp75 = [
# #     4.29,
# #     18.45,
# #     27.1,
# #     20.72,
# #     27.37,
# #     24.8,
# #     21.86,
# #     21.59,
# #     18.19,
# #     16.3,
# #     12.27,
# # ]
# topp50 = [
#     16.1,
#     34.45,
#     41.82,
#     93.85,
#     90.89,
#     98.08,
#     94.81,
#     92.36,
#     95.53,
#     96.74,
#     95.75,
# ]
# # topp50 = [
# #     1.02,
# #     1.8,
# #     3.84,
# #     0.41,
# #     0.27,
# #     0.36,
# #     0.11,
# #     0.23,
# #     0.13,
# #     0.07,
# #     0.51,
# # ]
# topk5 = [
#     26.49,
#     51.04,
#     90.11,
#     93.25,
#     93.69,
#     94.41,
#     95.36,
#     94.6,
#     95.96,
#     96,
#     96.26,
# ]
# # topk5 = [
# #     20.3,
# #     45.11,
# #     82.79,
# #     87.22,
# #     87.1,
# #     88.83,
# #     89.45,
# #     88.75,
# #     88.45,
# #     90.89,
# #     91.1,
# # ]
# topk3 = [
#     28.04,
#     71.63,
#     88.86,
#     94.67,
#     97.63,
#     99.26,
#     98.6,
#     98.66,
#     99.28,
#     99.4,
#     99.5,
# ]

# # topk3 = [
# #     5.76,
# #     24.68,
# #     40.66,
# #     45.02,
# #     52.96,
# #     51.34,
# #     48.78,
# #     53.48,
# #     50.04,
# #     54.37,
# #     54.12,
# # ]

# plt.scatter(range(len(sampling)), sampling, label="M.Sampling", marker="s")
# plt.scatter(range(len(topp9)), topp9, label="Top-p(0.9)", marker="v")
# plt.scatter(range(len(topp75)), topp75, label="Top-p(0.75)", marker="o")
# plt.scatter(range(len(topp50)), topp50, label="Top-p(0.5)", marker="X")
# plt.scatter(range(len(topk5)), topk5, label="Top-k(5)", marker="^")
# plt.scatter(range(len(topk3)), topk3, label="Top-k(3)", marker="D")
# # plt.plot(sampling, label="M.Sampling", marker="s")
# # plt.plot(topp9, label="Top-p(0.9)", marker="v")
# # plt.plot(topp75, label="Top-p(0.75)", marker="o")
# # plt.plot(topp50, label="Top-p(0.5)", marker="X")
# # plt.plot(topk5, label="Top-k(5)", marker="^")
# # plt.plot(topk3, label="Top-k(3)", marker="D")
# plt.xticks(range(len(sampling)))
# plt.xlabel("Iteration Steps")
# plt.tick_params(direction="in")
# plt.ylabel("Validity of generation (%)")
# # plt.title("Dataset Growth Over Iteration Steps")
# plt.legend()
# # plt.annotate("1000", (0, 1000), textcoords="offset points", xytext=(0, 10), ha="center")
# # last_points = [sampling[-1], topp9[-1], topp75[-1], topp50[-1], topk5[-1], topk3[-1]]
# # for i, txt in enumerate(last_points):
# #     if i == 1:
# #         plt.annotate(
# #             txt, (11, txt), textcoords="offset points", xytext=(1, 5), ha="center"
# #         )
# #     else:
# #         plt.annotate(
# #             txt, (11, txt), textcoords="offset points", xytext=(1, 1), ha="center"
# # )
# plt.show()


# # Decoding strategies in M2
# sampling = [
#     26,
#     106,
#     338,
#     816,
#     1484,
#     2247,
#     3025,
#     3810,
#     4594,
# ]

# topp9 = [
#     28,
#     171,
#     427,
#     784,
#     1076,
#     1355,
#     1616,
#     1824,
#     2041,
# ]

# topk5 = [
#     18,
#     91,
#     393,
#     1030,
#     1829,
#     2599,
#     3389,
#     4192,
#     4942,
# ]

# topk3 = [
#     13,
#     123,
#     292,
#     516,
#     835,
#     1166,
#     1545,
#     1913,
#     2233,
# ]
# sampling = [
#     0.3,
#     2.7,
#     7.7,
#     15.9,
#     22.3,
#     25.4,
#     25.9,
#     26.2,
#     26.1,
# ]
# topp9 = [
#     0.3,
#     4.8,
#     8.5,
#     11.9,
#     9.7,
#     9.3,
#     8.7,
#     6.9,
#     7.2,
# ]
# topk5 = [
#     0.2,
#     2.4,
#     10.1,
#     21.2,
#     26.6,
#     25.7,
#     26.3,
#     26.8,
#     25.0,
# ]
# topk3 = [
#     0.1,
#     3.7,
#     5.6,
#     7.5,
#     10.6,
#     11.0,
#     12.6,
#     12.3,
#     10.7,
# ]


# plt.scatter(range(len(sampling)), sampling, label="M.Sampling", marker="s")
# plt.scatter(range(len(topp9)), topp9, label="Top-p(0.9)", marker="v")
# plt.scatter(range(len(topk5)), topk5, label="Top-k(5)", marker="o")
# plt.scatter(range(len(topk3)), topk3, label="Top-k(3)", marker="X")
# plt.xticks(range(len(sampling)))
# # plt.plot(sampling, label="M.Sampling")
# # plt.plot(topp9, label="Top-p(0.9)")
# # plt.plot(topk5, label="Top-k(5)")
# # plt.plot(topk3, label="Top-k(3)")
# plt.xlabel("Iteration Steps")
# plt.tick_params(direction="in")
# plt.ylabel("Novelty of HP design generation(%)")
# plt.legend()
# plt.show()
