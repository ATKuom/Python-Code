import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3, suppress=True)
from scipy import optimize
from econ import economics
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
    e8 = m * ((h1 - h0) - (T0 + K) * (s1 - s0))  # W = kg/s*(J - °C*J/kgK)
    e9 = m * ((h2 - h0) - (T0 + K) * (s2 - s0))
    e10 = m * ((h3 - h0) - (T0 + K) * (s3 - s0))
    e11 = m * ((h4 - h0) - (T0 + K) * (s4 - s0))
    e12 = m * ((h5 - h0) - (T0 + K) * (s5 - s0))
    e7 = m * ((h6 - h0) - (T0 + K) * (s6 - s0))
    e_fgin = fg_m * ((hin_fg - h0_fg) - (T0 + K) * (sin_fg - s0_fg)) + 0.5e6
    e_fgout = fg_m * ((hout_fg - h0_fg) - (T0 + K) * (sout_fg - s0_fg)) + 0.5e6
    e_fuel = NG_exergy()

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
    cost_gt = 9.721e6  # $
    cost_motor = 211400 * (w_comp / 1e6) ** 0.6227
    cost_generator = (
        108900 * (w_tur * 0.95 / 1e6) ** 0.5463 + 177200 * (w_tur / 1e6) ** 0.2434
    )
    w_gt = 22.4e6
    pec = [
        2.916e6,
        1.944e6,
        3.888e6,
        0.972e6,
        cost_heater,  # 2.523e6,
        cost_tur,  # 2.734e6,
        cost_hx,  # 1.231e6,
        cost_comp,  # 1.939e6,
        cost_generator,
        cost_motor,
        cost_cooler,  # 1.327e6,
        (w_tur - w_comp) / 1e1,
    ]

    prod_capacity = (w_tur - w_comp + w_gt) / 1e6  # MW
    zk, cfueltot, lcoe = economics(pec, prod_capacity)  # $/h
    neg_zk = [-1 * i for i in zk]
    cfuel = 3.8e-9 * 3600  # cfueltot / e_fuel

    e1 = 0.08e6
    e2 = 27.62e6
    e3 = e_fuel
    e4 = 73.97e6
    e5 = e_fgin
    e6 = e_fgout
    e13 = 0.52e6
    e14 = 1.33e6
    e101 = 29.72e6
    e102 = 22.85e6
    e103 = w_comp
    e104 = w_tur
    e105 = w_gt
    e106 = w_comp
    e107 = 0.95 * w_tur
    e108 = e107 + e105 - e106
    m1 = np.array(
        [  # [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c101,c102,c103,c104,c105,c106,c107,c108]
            # GT compressor
            [e1, -e2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e101, 0, 0, 0, 0, 0, 0, 0],
            # GT combustor
            [0, e2, e3, -e4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # GT expander
            [
                0,
                0,
                0,
                e4,
                -e5,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -e101,
                -e102,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
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
    )
    m2 = np.asarray(
        neg_zk[:-2]
        + [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            cfuel,
            0,
            0,
        ]
    ).reshape(
        -1,
    )
    try:
        # costs1, _, _, _ = np.linalg.lstsq(m1, m2, rcond=None)  # $/Wh
        # costs2, _ = optimize.nnls(m1, m2[:, 0])
        # print(costs1 / 3600 * 1e9)
        # print(costs2 / 3600 * 1e9)
        costs = np.linalg.solve(m1, m2)

    except:
        return PENALTY_VALUE

    """
    fuel_chem_ex = 1.26/16.043*824.348  # MW = kg/s /kg/kmol *MJ/kmol
    fuel_phys_ex = 1.26*(0.39758) #MW = kg/s * MJ/kg
    Efuel = fuel_chem_ex + fuel_phys_ex  # MW
    Cp=cfuel*Efuel  + Ztot # $/h
    Ep = 22.4 + w_tur/1e6 - w_comp/1e6 # MW
    cdiss = c2*e2 - c3*e3 + zk[2] # $/h = $/Wh * W - $/Wh * W + $/h
    Cp = 8700 * (q_heater / 1e6) * 3600 + Ztot  # $/h = $/MJ * MJ/s * s/h + $/h
    Ep = (w_tur - w_comp) / 1e6  # MW
    """
    Cl = costs[5] * e_fgout  # $/h
    Cf = costs[2] * e_fuel  # $/h
    Ztot = sum(zk)  # $/h
    Cp = Cf + Ztot - Cl  # $/h
    Ep = e108  # W
    # c = Cp / Ep  # $/MWh
    cdiss = (e9 * costs[8] - e10 * costs[9]) + zk[-2]
    lcoex = (costs[21] * Ep + cdiss + Cl) / (Ep / 1e6)
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
    {costs/3600*1e9}
    {pec}
    {zk}
    {sum(pec)}
    {sum(zk)}
    {cdiss, Cl, costs[21] * Ep, lcoex,lcoe}
    {Cp/(Ep/1e6)}
    """
    )

    # Turbine Pratio = {tur_pratio:.2f}   p6/p1={Pressure[5]:.2f}bar/{Pressure[0]:.2f}bar
    # Turbine output = {unit_energy[0]:.2f}MW
    # Compressor Pratio = {comp_pratio:.2f}   p3/p4={Pressure[3]:.2f}bar/{Pressure[2]:.2f}bar
    # Compressor Input = {unit_energy[1]:.2f}MW
    # Temperatures = t1={t1:.1f}   t2={t2:.1f}    t3={t3:.1f}    t4={t4:.1f}     t5={t5:.1f}    t6={t6:.1f}   Tstack={fg_tout:.1f}    DT ={approach_temp:.1f}
    # Pressures =    p1={Pressure[0]:.1f}bar p2={Pressure[1]:.1f}bar p3={Pressure[2]:.1f}bar p4={Pressure[3]:.1f}bar p5={Pressure[4]:.1f}bar p6={Pressure[5]:.1f}bar
    # Equipment Cost = Tur={cost_tur:.0f}    HX={cost_hx:.0f}    Cooler={cost_cooler:.0f}    Compr={cost_comp:.0f}   Heater={cost_heater:.0f}
    # Equipment Energy = Qheater={unit_energy[2]:.2f}MW  Qcooler={unit_energy[3]:.2f}MW  Qhx={unit_energy[4]:.2f}MW
    # Objective Function value = {c}
    # Exergy of streams = {e1/1e6:.2f}MW {e2/1e6:.2f}MW {e3/1e6:.2f}MW {e4/1e6:.2f}MW {e5/1e6:.2f}MW {e6/1e6:.2f}MW {e_fgin/1e6 :.2f}MW {e_fgout/1e6:.2f}MW
    # {costs/3600*1e9}
    # {pec}
    # {zk}
    # {sum(zk)}
    # {sum(pec)}
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
    x2 = [
        32,
        427.14871102310946,
        7838558.624054214,
        30000000.0,
        74.68430647939505,
        10.7,
    ]
    result_analyses(x1)
