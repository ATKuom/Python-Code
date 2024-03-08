import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3, suppress=True)
from scipy import optimize
from First_year_version.econ import economics
from functions import (
    lmtd,
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

    # Turbine
    h1, s1, t1, w_tur = turbine(t6, p6, p1, ntur, m)
    ##Compressor
    h4, s4, t4, w_comp = compressor(t3, p3, p4, ncomp, m)
    ##Heat Exchanger
    t2, h2, s2, t5, h5, s5, q_hx = HX_calculation(
        t1, p1, h1, t4, p4, h4, approach_temp, hx_pdrop, m
    )
    ##Cooler
    h3, s3, q_cooler = cooler(t2, p2, t3, cooler_pdrop, m)
    ##Heater
    h6, s6, q_heater = heater(t5, p5, t6, heater_pdrop, m)
    fg_tout = fg_calculation(fg_m, q_heater)
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
    UA_cooler = (q_cooler / 1) / lmtd(dt1_cooler, dt2_cooler)  # W / °C
    if t2 > 550:
        ft_cooler = 1 + 0.02141 * (t2 - 550)
    else:
        ft_cooler = 1
    cost_cooler = 49.45 * UA_cooler**0.7544 * ft_cooler  # $

    cost_comp = 1230000 * (w_comp / 1e6) ** 0.3992  # $

    dt1_heater = fg_tin - t6  # °C
    dt2_heater = fg_tout - t5  # °C
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
    pec.append(cost_heater)
    pec.append(cost_comp)
    pec.append(cost_cooler)
    prod_capacity = (w_tur - w_comp) / 1e6  # MW
    zk, cfuel, lcoe = economics(pec, prod_capacity)  # $/h

    m1 = np.array(
        [  # [c1,c2,c3,c4,c5,c6,cwt,cwcomp,ctote,cfgin,cfgout]
            # Turbine
            [e1, 0, 0, 0, 0, -e6, w_tur, 0, 0, 0, 0],
            # HXer
            [-e1, e2, 0, -e4, e5, 0, 0, 0, 0, 0, 0],
            # Heater
            [0, 0, 0, 0, -e5, e6, 0, 0, 0, -e_fgin, e_fgout],
            # Compressor
            [0, 0, -e3, e4, 0, 0, 0, -w_comp, 0, 0, 0],
            # [0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0],
            # Turbine aux1
            [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            # HXer aux1
            [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Cost of FG
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            # Cooler aux1
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            # Heater aux1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
            # Total electricity production
            [0, 0, 0, 0, 0, 0, w_tur, -w_comp, -(w_tur - w_comp), 0, 0],
            # Total electricity aux1
            [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
        ]
    )
    m2 = np.asarray(zk[:4] + [0, 0, 8.9e-9 * 3600, 0, 0, 0, 0]).reshape(
        -1,
    )

    try:
        costs = np.linalg.solve(m1, m2)
    except:
        return PENALTY_VALUE

    Cl = costs[10] * e_fgout  # $/h
    Cf = costs[9] * e_fgin  # $/h
    Ztot = sum(zk)  # $/h
    Cp = Cf + Ztot - Cl  # $/h
    Ep = w_tur - w_comp  # MW
    cdiss = costs[1] * e2 - costs[2] * e3 + zk[-1]
    lcoex = (costs[-3] * Ep + cdiss + Cl) / (Ep / 1e6)
    c = lcoex
    thermal_efficiency = (w_tur - w_comp) / 40.53e6
    if thermal_efficiency < 0.1575:
        j = 1e5 * (0.30 - thermal_efficiency)
    else:
        j = c + 1e2 * max(0, 0.95 - q_hx / q_heater)
    Pressure = [p1 / 1e5, p2 / 1e5, p3 / 1e5, p4 / 1e5, p5 / 1e5, p6 / 1e5]
    unit_energy = [
        w_tur / 1e6,
        w_comp / 1e6,
        q_heater / 1e6,
        q_cooler / 1e6,
        q_hx / 1e6,
    ]
    np.set_printoptions(precision=2, suppress=True)
    print(
        f"""
    Turbine Pratio = {tur_pratio:.2f}   p6/p1={Pressure[5]:.2f}bar/{Pressure[0]:.2f}bar
    Turbine output = {unit_energy[0]:.2f}MW
    Compressor Pratio = {comp_pratio:.2f}   p3/p4={Pressure[3]:.2f}bar/{Pressure[2]:.2f}bar
    Compressor Input = {unit_energy[1]:.2f}MW
    Temperatures = t1={t1:.1f}   t2={t2:.1f}    t3={t3:.1f}    t4={t4:.1f}     t5={t5:.1f}    t6={t6:.1f}   Tstack={fg_tout:.1f}    DT ={approach_temp:.1f}
    Pressures =    p1={Pressure[0]:.1f}bar p2={Pressure[1]:.1f}bar p3={Pressure[2]:.1f}bar p4={Pressure[3]:.1f}bar p5={Pressure[4]:.1f}bar p6={Pressure[5]:.1f}bar
    Equipment Cost = Tur={cost_tur/1e3:.0f}    HX={cost_hx/1e3:.0f}    Cooler={cost_cooler/1e3:.0f}    Compr={cost_comp/1e3:.0f}   Heater={cost_heater/1e3:.0f}
    Equipment Energy = Qheater={unit_energy[2]:.2f}MW  Qcooler={unit_energy[3]:.2f}MW  Qhx={unit_energy[4]:.2f}MW
    Objective Function value = {c}
    Exergy of streams = {e1/1e6:.2f}MW {e2/1e6:.2f}MW {e3/1e6:.2f}MW {e4/1e6:.2f}MW {e5/1e6:.2f}MW {e6/1e6:.2f}MW {e_fgin/1e6:.2f}MW {e_fgout/1e6:.2f}MW
    Exergy costing of streams = {costs/3600*1e9} $/GJ
    Total PEC = {sum(pec):.2f} $
    Total Zk  = {sum(zk):.2f} $/h
    Cdiss = {cdiss:.2f} Cl = {Cl:.2f} Cp ={costs[-3]*Ep:.2f} LCOE = {lcoe:.2f} LCOEX = {lcoex:.2f}
    Cp/Ep = {Cp/(Ep/1e6):.2f}
    Thermal efficiency = {Ep/40.53e6*100:.2f}%
    j = {j:.2f}
    Heat recuperation ratio = {q_hx/q_heater:.2f}
        """
    )

    return c


if __name__ == "__main__":
    x1 = [
        32.3,
        411.4,
        78.5e5,
        241.3e5,
        93.18,
        10.8,
    ]
    # PSO_1a using c=lcoe
    a1 = [
        32,
        421.07187779100747,
        7804322.47429322,
        30000000.0,
        84.27474563424512,
        4.0168474911447545,
    ]
    fit1 = 192.95038653511196
    # PSO_1a using c=lcoex
    a2 = [32, 430.5046261817056, 7816866.407857994, 30000000.0, 70.36707712421733, 11]
    fit2 = 97.87292554166655
    x = [
        32.0,
        355.64284972779393,
        7898527.438710488,
        30000000.0,
        105.99196786493403,
        11.0,
    ]
    result_analyses(x1)
