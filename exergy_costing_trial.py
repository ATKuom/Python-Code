import numpy as np
import scipy.optimize as optimize

e1 = 0.08e6
e2 = 27.62e6
e3 = 65.4e6
e4 = 73.97e6
e5 = 18.02e6
e6 = 4.43e6
e7 = 40.08e6
e8 = 28.98e6
e9 = 20.39e6
e10 = 18.77e6
e11 = 21.12e6
e12 = 27.79e6
e13 = 0.52e6
e14 = 1.33e6
e101 = 29.72e6
e102 = 22.85e6
e103 = 2.77e6
e104 = 10.20e6
e105 = 22.4e6
e106 = 2.77e6
e107 = 9.69e6
e108 = 29.3126e6
m1 = np.array(
    [  # [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c101,c102,c103,c104,c105,c106,c107,c108]
        # GT compressor
        [e1, -e2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e101, 0, 0, 0, 0, 0, 0, 0],
        # GT combustor
        [0, e2, e3, -e4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # GT expander
        [0, 0, 0, e4, -e5, 0, 0, 0, 0, 0, 0, 0, 0, 0, -e101, -e102, 0, 0, 0, 0, 0, 0],
        # GT generator
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e102, 0, 0, -e105, 0, 0, 0],
        # Primary heater
        [0, 0, 0, 0, e5, -e6, -e7, 0, 0, 0, 0, e12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # SCO2 expander
        [0, 0, 0, 0, 0, 0, e7, -e8, 0, 0, 0, 0, 0, 0, 0, 0, 0, -e104, 0, 0, 0, 0],
        # Recuperator
        [0, 0, 0, 0, 0, 0, 0, e8, -e9, 0, e11, -e12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Cooler
        # [0, 0, 0, 0, 0, 0, 0, 0, e9, -e10, 0, 0, e13, -e14, 0, 0, 0, 0, 0, 0, 0, 0],
        # SCO2 compressor
        [0, 0, 0, 0, 0, 0, 0, 0, 0, e10, -e11, 0, 0, 0, 0, 0, e103, 0, 0, 0, 0, 0],
        # SCO2 generator
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e104, 0, 0, -e107, 0],
        # SCO2 motor
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -e103, 0, 0, e106, 0, 0],
        # Power summarizer
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            e105,
            -e106,
            e107,
            -e108,
        ],
        # GT expander aux1
        [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # GT expander aux2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        # Primary heater aux1
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # SCO2 expander aux1
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Recuperator aux1
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Cooler aux1
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Cooler aux2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        # Power summarizer aux1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],
        # Fuel cost
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Air cost
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Cooling water cost
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)  # W
m1 = m1
x = np.array(
    [
        0,
        13.6,
        3.8,
        8.9,
        8.9,
        8.9,
        21.9,
        21.9,
        21.9,
        21.9,
        23.7,
        25.6,
        0,
        0,
        10.8,
        10.8,
        20.3,
        28.4,
        11.8,
        17.7,
        31.2,
        17.7,
    ]
)
x = x * 3600 / 1e9
m2 = np.array(
    [
        -191.31,
        -127.54,
        -255.08,
        -63.77,
        -165.54,
        -170.85,
        -76.94,
        # 124.7,
        -121.17,
        -44.91,
        -26.15,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3.8 * 3600 / 1e9,
        0,
        0,
    ]
)
np.set_printoptions(precision=2, suppress=True)
# c1 = m1 @ x
# print(c1)
# c, _, _, _ = np.linalg.lstsq(m1, m2, rcond=None)
# costs = np.linalg.solve(m1, m2)
# costs2, _ = optimize.nnls(m1, m2)
# print(c * 1e9 / 3600)
# print(costs2 * 1e9 / 3600)
# print(costs * 1e9 / 3600)
m1 = np.array(
    [  # [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c101,c102,c103,c104,c105,c106,c107,c108]
        # GT
        [e3, -e5, 0, 0, 0, 0, 0, 0, 0, 0, 0, -e105, 0, 0, 0],
        # Primary heater
        [0, e5, -e6, -e7, 0, 0, 0, 0, e12, 0, 0, 0, 0, 0, 0],
        # SCO2 expander
        [0, 0, 0, e7, -e8, 0, 0, 0, 0, 0, 0, 0, 0, -e107, 0],
        # Recuperator
        [0, 0, 0, 0, e8, -e9, 0, e11, -e12, 0, 0, 0, 0, 0, 0],
        # Cooler
        # [0, 0, 0, 0, 0, 0, 0, 0, e9, -e10, 0, 0, e13, -e14, 0, 0, 0, 0, 0, 0, 0, 0],
        # SCO2 compressor
        [0, 0, 0, 0, 0, 0, e10, -e11, 0, 0, 0, 0, e106, 0, 0],
        # Power summarizer
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e105, -e106, e107, -e108],
        # Primary heater aux1
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # SCO2 expander aux1
        [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Recuperator aux1
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Cooler aux1
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        # Cooler aux2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        # Power summarizer aux1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],
        # Fuel cost
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Cooling water cost
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    ]
)  # W

m2 = np.array(
    [
        -191.31 - 127.54 - 255.08 - 63.77,
        -165.54,
        -170.85 - 44.91,
        -76.94,
        # 124.7,
        -121.17 - 26.15,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3.8 * 3600 / 1e9,
        0,
    ]
)
# print(m1.shape)
# costs2, _ = optimize.nnls(m1, m2)
# print(costs2 * 1e9 / 3600)

m1 = np.array(
    [  # [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c101,c102,c103,c104,c105,c106,c107,c108]
        # GT compressor
        [e1, -e2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e101, 0, 0, 0, 0, 0, 0, 0],
        # GT combustor
        [0, e2, e3, -e4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # GT expander
        [0, 0, 0, e4, -e5, 0, 0, 0, 0, 0, 0, 0, 0, 0, -e101, -e102, 0, 0, 0, 0, 0, 0],
        # GT generator
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e102, 0, 0, -e105, 0, 0, 0],
        # GT expander aux1
        [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # GT expander aux2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        # Fuel cost
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Air cost
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)  # W
print(m1.shape)
m2 = np.array(
    [
        -191.31,
        -127.54,
        -255.08,
        -63.77,
        0,
        0,
        3.8 * 3600 / 1e9,
        0,
    ]
)
costs2, _ = optimize.nnls(m1, m2)
print(costs2 * 1e9 / 3600)