import random
import matplotlib.pyplot as plt
import numpy as np
from New_RS import result_analyses
from First_year_version.econ import economics
from functions import (
    pinch_calculation,
    lmtd,
    enthalpy_entropy,
    turbine,
    compressor,
    cooler,
    heater,
    h_s_fg,
    fg_calculation,
    HX_calculation,
    cw_Tout,
    NG_exergy,
    h0_fg,
    s0_fg,
    hin_fg,
    sin_fg,
    h0,
    s0,
    T0,
    P0,
    K,
)

bounds = [
    (32, 38),
    (180, 530),
    (74e5, 300e5),
    (74e5, 300e5),
    (50, 160),
    (4, 10.7),
]  # upper and lower bounds of variables

# PARAMETERS OF PSO
particle_size = 1  # number of particles
iterations = 1  # max number of iterations
nv = len(bounds)  # number of variables


# ------------------------------------------------------------------------------
def result_analyses(x):
    t3 = x[0]
    t6 = x[1]
    p1 = x[2]
    p4 = x[3]
    m = x[4]
    approach_temp = x[5]

    ##Parameters
    ntur = 85  # turbine efficiency     2019 Nabil
    ncomp = 82  # compressor efficiency 2019 Nabil
    cw_temp = 19  # °C
    fg_tin = 539  # °C
    fg_m = 68.75  # kg/s
    cooler_pdrop = 1e5
    heater_pdrop = 0
    hx_pdrop = 0.5e5
    PENALTY_VALUE = float(1e6)
    pec = list()

    p2 = p1 - hx_pdrop
    p3 = p2 - cooler_pdrop
    p5 = p4 - hx_pdrop
    p6 = p5 - heater_pdrop
    tur_pratio = p6 / p1
    comp_pratio = p4 / p3
    if tur_pratio < 1 or comp_pratio < 1:
        return PENALTY_VALUE
    # Turbine
    h1, s1, t1, w_tur = turbine(t6, p6, p1, ntur, m)
    if w_tur < 0:
        # print("negative turbine work")
        return PENALTY_VALUE

    ##Compressor
    h4, s4, t4, w_comp = compressor(t3, p3, p4, ncomp, m)
    if w_comp > w_tur:
        # print("negative net power production")
        return PENALTY_VALUE

    ##Heat Exchanger
    t2, h2, s2, t5, h5, s5, q_hx = HX_calculation(
        t1, p1, h1, t4, p4, h4, approach_temp, hx_pdrop, m
    )
    if t2 == 0:
        return PENALTY_VALUE
    ##Cooler
    if t3 > t2:
        # print("negative cooler work")
        return PENALTY_VALUE
    h3, s3, q_cooler = cooler(t2, p2, t3, cooler_pdrop, m)

    ##Heater
    h6, s6, q_heater = heater(t5, p5, t6, heater_pdrop, m)
    fg_tout = fg_calculation(fg_m, q_heater)
    if fg_tout < 90:
        # print("too low flue gas stack temperature")
        return PENALTY_VALUE
    hout_fg, sout_fg = h_s_fg(fg_tout, P0)

    # Exergy Analysis
    e1 = m * ((h1 - h0) - (T0 + K) * (s1 - s0))  # W = kg/s*(J - °C*J/kgK)
    e2 = m * ((h2 - h0) - (T0 + K) * (s2 - s0))
    e3 = m * ((h3 - h0) - (T0 + K) * (s3 - s0))
    e4 = m * ((h4 - h0) - (T0 + K) * (s4 - s0))
    e5 = m * ((h5 - h0) - (T0 + K) * (s5 - s0))
    e6 = m * ((h6 - h0) - (T0 + K) * (s6 - s0))
    e_fgin = fg_m * ((hin_fg - h0_fg) - (T0 + K) * (sin_fg - s0_fg)) + 0.5e6
    e_fgout = fg_m * ((hout_fg - h0_fg) - (T0 + K) * (sout_fg - s0_fg)) + 0.5e6

    # Economic Analysis
    if t6 > 550:
        ft_tur = 1 + 1.137e-5 * (t6 - 550) ** 2
    else:
        ft_tur = 1
    cost_tur = 406200 * ((w_tur / 1e6) ** 0.8) * ft_tur  # $

    dt1_cooler = t3 - cw_temp  # °C
    dt2_cooler = t2 - cw_Tout(q_cooler)  # °C
    if dt2_cooler < 0 or dt1_cooler < 0:
        return PENALTY_VALUE
    UA_cooler = (q_cooler / 1) / lmtd(dt1_cooler, dt2_cooler)  # W / °C
    if t2 > 550:
        ft_cooler = 1 + 0.02141 * (t2 - 550)
    else:
        ft_cooler = 1
    cost_cooler = 49.45 * UA_cooler**0.7544 * ft_cooler  # $

    cost_comp = 1230000 * (w_comp / 1e6) ** 0.3992  # $

    dt1_heater = fg_tin - t6  # °C
    dt2_heater = fg_tout - t5  # °C
    if dt2_heater < 0 or dt1_heater < 0:
        return PENALTY_VALUE
    UA_heater = (q_heater / 1e3) / lmtd(dt1_heater, dt2_heater)  # W / °C
    cost_heater = 5000 * UA_heater  # Thesis 97/pdf116

    dt1_hx = t1 - t5  # °C
    dt2_hx = t2 - t4  # °C
    UA_hx = (q_hx / 1) / lmtd(dt1_hx, dt2_hx)  # W / °C
    if t1 > 550:
        ft_hx = 1 + 0.02141 * (t1 - 550)
    else:
        ft_hx = 1
    cost_hx = 49.45 * UA_hx**0.7544 * ft_hx  # $
    pec.append(cost_tur)
    pec.append(cost_hx)
    pec.append(cost_cooler)
    pec.append(cost_heater)
    pec.append(cost_comp)
    prod_capacity = (w_tur - w_comp) / 1e6  # MW
    zk, cfuel, lcoe = economics(pec, prod_capacity)  # $/h
    # [c1,c2,c3,c4,c5,c6,cw]
    # m1 = np.array(
    #     [
    #         [e1, 0, 0, 0, 0, -e6, w_tur, 0],
    #         [e1, e2, 0, -e4, e5, 0, 0, 0],
    #         [0, e2, -e3, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, -e5, e6, 0, -(e_exin - e_exout)],
    #         [0, 0, -e3, e4, 0, 0, -w_comp, 0],
    #         [1, 0, 0, 0, 0, -1, 0, 0],
    #         [1, -1, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 1],
    #     ]
    # )  # W
    # m2 = np.asarray(zk + [0, 0, 8.7e-9 * 3600]).reshape(8, 1)
    # try:
    #     costs = np.linalg.solve(m1, m2)  # $/Wh
    #     print(costs)
    # except:
    #     return PENALTY_VALUE
    """
    fuel_chem_ex = 1.26/16.043*824.348  # MW = kg/s /kg/kmol *MJ/kmol
    fuel_phys_ex = 1.26*(0.39758) #MW = kg/s * MJ/kg
    Efuel = fuel_chem_ex + fuel_phys_ex  # MW
    Cp=cfuel*Efuel  + Ztot # $/h
    Ep = 22.4 + w_tur/1e6 - w_comp/1e6 # MW
    Cdiss = c2*e2 - c3*e3 + zk[2] # $/h = $/Wh * W - $/Wh * W + $/h
    Cp = 8700 * (q_heater / 1e6) * 3600 + Ztot  # $/h = $/MJ * MJ/s * s/h + $/h
    Ep = (w_tur - w_comp) / 1e6  # MW
    """

    # Cl = costs[7] * e_exout  # $/h
    # Cf = costs[7] * e_exin  # $/h
    # Ztot = sum(zk)  # $/h
    # Cp = Cf + Ztot - Cl  # $/h
    # Ep = (w_tur - w_comp) / 1e6 + 22.4  # MW
    # c = Cp / Ep  # $/MWh
    c = lcoe
    Pressure = [p1 / 1e5, p2 / 1e5, p3 / 1e5, p4 / 1e5, p5 / 1e5, p6 / 1e5]
    unit_energy = [
        w_tur / 1e6,
        w_comp / 1e6,
        q_heater / 1e6,
        q_cooler / 1e6,
        q_hx / 1e6,
    ]
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
        Temperatures = t1={t1:.0f}  t2={t2:.0f} t3={t3:.0f} t4={t4:.0f} t5={t5:.0f} t6={t6:.0f} Tstack = {fg_tout:.0f}
        Equipment Cost = Tur={cost_tur:.0f}    HX={cost_hx:.0f}    Cooler={cost_cooler:.0f}    Compr={cost_comp:.0f}   Heater={cost_heater:.0f}
        Equipment Energy = Qheater={unit_energy[2]:.2f}MW  Qcooler={unit_energy[3]:.2f}MW  Qhx={unit_energy[4]:.2f}MW
        Exergy of streams = {e1/1e6:.2f}MW {e2/1e6:.2f}MW {e3/1e6:.2f}MW {e4/1e6:.2f}MW {e5/1e6:.2f}MW {e6/1e6:.2f}MW
        Ep = 
        Objective Function value = {c}
        
        """
    )

    return c


if __name__ == "__main__":
    x = [32, 387.8753460810269, 7953215.395366016, 30000000.0, 95.50484829406095, 4]
    result_analyses(x)
