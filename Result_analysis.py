import random
import matplotlib.pyplot as plt
import numpy as np
from functions import (
    Pressure_calculation,
    pinch_calculation,
    specific_heat,
    temperature,
    lmtd,
    enthalpy_entropy,
    h0,
    s0,
    T0,
    K,
)
from econ import economics


# ------------------------------------------------------------------------------
def result_analyses(x):
    t3 = x[0]
    t6 = x[1]
    tur_pratio = x[2]
    comp_pratio = x[3]
    m = x[4]

    ##Parameters
    ntur = 0.93  # turbine efficiency     2019 Nabil
    ncomp = 0.89  # compressor efficiency 2019 Nabil
    gamma = 1.28  # 1.28 or 1.33 can be used based on the assumption
    air_temp = 15
    exhaust_Tin = 630  # °C
    exhaust_m = 935  # kg/s
    cp_gas = 1151  # j/kgK
    PENALTY_VALUE = float(1e9)
    pec = list()

    p1, p2, p3, p4, p5, p6 = Pressure_calculation(tur_pratio, comp_pratio)
    if p1 == 0:
        return PENALTY_VALUE

    # Turbine
    (h6, s6) = enthalpy_entropy(t6, p6)
    t1 = (t6 + K) - ntur * ((t6 + K) - (t6 + K) / (tur_pratio ** (1 - 1 / gamma))) - K
    (h1, s1) = enthalpy_entropy(t1, p1)
    w_tur = m * (h6 - h1)

    if w_tur < 0:
        return PENALTY_VALUE

    ##Compressor
    (h3, s3) = enthalpy_entropy(t3, p3)
    t4 = (t3 + K) + ((t3 + K) * (comp_pratio ** (1 - 1 / gamma)) - (t3 + K)) / ncomp - K
    (h4, s4) = enthalpy_entropy(t4, p4)
    w_comp = m * (h4 - h3)

    if w_comp < 0:
        return PENALTY_VALUE

    ##Heat Exchanger
    t2, t5 = pinch_calculation(t1, h1, t4, h4, p2, p5, m)
    (h2, s2) = enthalpy_entropy(t2, p2)
    q_hx = m * (h1 - h2)

    ##Cooler
    if t3 > t2:
        return PENALTY_VALUE
    q_c = m * (h2 - h3)

    ##Heater
    (h5, s5) = enthalpy_entropy(t5, p5)
    q_heater = m * (h6 - h5)
    exhaust_Tout = exhaust_Tin - q_heater / (exhaust_m * cp_gas)

    e1 = m * (h1 - h0 - (T0 + K) * (s1 - s0))
    e2 = m * (h2 - h0 - (T0 + K) * (s2 - s0))
    e3 = m * (h3 - h0 - (T0 + K) * (s3 - s0))
    e4 = m * (h4 - h0 - (T0 + K) * (s4 - s0))
    e5 = m * (h5 - h0 - (T0 + K) * (s5 - s0))
    e6 = m * (h6 - h0 - (T0 + K) * (s6 - s0))

    # Economic Analysis

    if t6 > 550:
        ft_tur = 1 + 1.106e-4 * (t6 - 550) ** 2
    else:
        ft_tur = 1
    cost_tur = 182600 * ((w_tur / 1e6) ** 0.5561) * ft_tur

    dt1_cooler = t2 - air_temp
    dt2_cooler = t3 - air_temp
    UA_cooler = q_c / lmtd(dt1_cooler, dt2_cooler)
    cost_cooler = 32.88 * UA_cooler**0.75

    cost_comp = 1230000 * (w_comp / 1e6) ** 0.3992

    dt1_heater = exhaust_Tin - t6
    dt2_heater = exhaust_Tout - t5
    UA_heater = q_heater / lmtd(dt1_heater, dt2_heater)
    if t6 > 550:
        ft_heater = 1 + 0.02141 * (t6 - 550)
    else:
        ft_heater = 1
    cost_heater = 49.45 * UA_heater**0.7544 * ft_heater

    dt1_hx = t1 - t5
    dt2_hx = t2 - t4
    UA_hx = q_hx / lmtd(dt1_hx, dt2_hx)
    if t1 > 550:
        ft_hx = 1 + 0.02141 * (t1 - 550)
    else:
        ft_hx = 1
    cost_hx = 49.45 * UA_hx**0.7544 * ft_hx
    pec.append(cost_tur)
    pec.append(cost_hx)
    pec.append(cost_cooler)
    pec.append(cost_heater)
    pec.append(cost_comp)
    prod_capacity = (w_tur - w_comp) / 1e6
    if w_tur < w_comp:
        return PENALTY_VALUE
    zk, cftot = economics(pec, prod_capacity)
    # [c1,c2,c3,c4,c5,c6,cw]
    m1 = np.array(
        [
            [e1, 0, 0, 0, 0, -e6, w_tur],
            [e1, e2, 0, -e4, e5, 0, 0],
            [0, e2, -e3, 0, 0, 0, 0],
            [0, 0, 0, 0, -e5, e6, 0],
            [0, 0, -e3, e4, 0, 0, -w_comp],
            [1, 0, 0, 0, 0, -1, 0],
            [1, -1, 0, 0, 0, 0, 0],
        ]
    )
    m2 = np.asarray(zk + [0, 0]).reshape(7, 1)
    costs = np.linalg.solve(m1, m2)
    Cp = costs[6] * w_tur + costs[1] * e2 + costs[5] * e6 - 2 * costs[2] * e3
    Cf = cftot * q_heater + costs[6] * w_comp + costs[5] * e6 - costs[1] * e2
    Ztot = sum(zk)
    Cl = Cf - Cp - Ztot
    Ep = (w_tur + e2 + e6 + -2 * e3) / 1e6
    c = Cp / Ep
    Pressure = [p1 / 1e5, p2 / 1e5, p3 / 1e5, p4 / 1e5, p5 / 1e5, p6 / 1e5]
    unit_energy = [w_tur / 1e6, w_comp / 1e6, q_heater / 1e6, q_c / 1e6, q_hx / 1e6]
    print(
        f"""
        p6 = {Pressure[5]:.2f} bar
        p1 = {Pressure[0]:.2f} bar
        Turbine Pratio = {tur_pratio:.2f}
        Turbine output = {unit_energy[0]:.2f} MW
        p3 = {Pressure[2]:.2f} bar
        p4 = {Pressure[3]:.2f} bar
        Compressor Pratio = {comp_pratio:.2f}
        Compressor Input = {unit_energy[1]:.2f} MW
        Temperatures = t1={t1:.0f}  t2={t2:.0f} t3={t3:.0f} t4={t4:.0f} t5={t5:.0f} t6={t6:.0f}
        Equipment Cost = Tur={cost_tur:.0f}    HX={cost_hx:.0f}    Cooler={cost_cooler:.0f}    Compr={cost_comp:.0f}   Heater={cost_heater:.0f}
        Exergy of streams = {e1/1e6:.2f}MW {e2/1e6:.2f}MW {e3/1e6:.2f}MW {e4/1e6:.2f}MW {e5/1e6:.2f}MW {e6/1e6:.2f}MW
        Ef = {e6-e1, e1-e2,q_c,q_heater,w_comp}
        Ep = 
        Objective Function value = {c}
        
        """
    )
    return c


if __name__ == "__main__":
    x = [436.574327553335, 557.9441293295541, 1, 1.0271463264317666, 200]
    result_analyses(x)
