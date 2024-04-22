import matplotlib.pyplot as plt
import numpy as np
from pyfluids import Fluid, FluidsList, Input, Mixture

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
    split_ratio = x[0]
    t2 = x[1]
    approach_temp = x[2]
    mixer = x[3]
    t6 = x[4]
    p7 = x[5]
    t9 = x[6]
    p10 = x[7]
    m_set = x[8]

    ##Parameters
    ntur = 85  # turbine efficiency     2019 Nabil
    ncomp = 82  # compressor efficiency 2019 Nabil
    cw_temp = 19  # °C
    fg_tin = 539.76  # °C
    fg_m = 68.75  # kg/s
    cooler_pdrop = 1e5
    heater_pdrop = 0
    hx_pdrop = 0.5e5
    PENALTY_VALUE = float(1e6)
    pec = list()

    m = np.ones(10) * m_set
    m[0] = split_ratio * m_set
    m[1] = m[0]
    m[2] = (1 - split_ratio) * m_set
    m[3] = m[2]

    p8 = p7 - hx_pdrop
    p9 = p8 - cooler_pdrop
    p1 = p10
    p2 = p1 - heater_pdrop
    p3 = p1
    p4 = p3 - hx_pdrop
    p5 = min(p4, p2)
    p6 = p5 - heater_pdrop
    tur_pratio = p6 / p7
    comp_pratio = p10 / p9

    ##Turbine
    h7, s7, t7, w_tur = turbine(t6, p6, p7, ntur, m[6])
    ##Compressor
    h10, s10, t10, w_comp = compressor(t9, p9, p10, ncomp, m[9])
    ##Splitter
    h1, s1, t1 = h10, s10, t10
    ##Heater1
    h2, s2, q_heater1 = heater(t1, p1, t2, heater_pdrop, m[1])
    ##Mixer1
    h3, s3, t3 = h1, s1, t1
    ##Heat Exchanger
    t8, h8, s8, t4, h4, s4, q_hx = HX_calculation(
        t7, p7, h7, t3, p3, h3, approach_temp, hx_pdrop, m[7], m[3]
    )
    ##Mixer2
    if p4 == p2:
        inlet1 = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(t2),
            Input.pressure(p2),
        )
        inlet2 = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(t4),
            Input.pressure(p4),
        )
        outlet = Fluid(FluidsList.CarbonDioxide).mixing(m[1], inlet1, m[3], inlet2)
        t5 = outlet.temperature
        h5 = outlet.enthalpy
        s5 = outlet.entropy
    if p2 > p4:
        hp_inlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(
                Input.temperature(t2),
                Input.pressure(p2),
            )
            .isenthalpic_expansion_to_pressure(p4)
        )
        lp_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(t4),
            Input.pressure(p4),
        )
        outlet = Fluid(FluidsList.CarbonDioxide).mixing(m[1], hp_inlet, m[3], lp_inlet)
        t5 = outlet.temperature
        h5 = outlet.enthalpy
        s5 = outlet.entropy
    else:
        hp_inlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(
                Input.temperature(t4),
                Input.pressure(p4),
            )
            .isenthalpic_expansion_to_pressure(p2)
        )
        lp_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(t2),
            Input.pressure(p2),
        )
        outlet = Fluid(FluidsList.CarbonDioxide).mixing(m[3], hp_inlet, m[1], lp_inlet)
        t5 = outlet.temperature
        h5 = outlet.enthalpy
        s5 = outlet.entropy
    ##Heater2
    h6, s6, q_heater2 = heater(t5, p5, t6, heater_pdrop, m[5])
    ##Cooler
    h9, s9, q_cooler = cooler(t8, p8, t9, cooler_pdrop, m[8])

    fg_tout = fg_calculation(fg_m, q_heater1 + q_heater2)

    # Exergy Analysis
    e1 = m[0] * ((h1 - h0) - (T0 + K) * (s1 - s0))  # W = kg/s*(J - °C*J/kgK)
    e2 = m[1] * ((h2 - h0) - (T0 + K) * (s2 - s0))
    e3 = m[2] * ((h3 - h0) - (T0 + K) * (s3 - s0))
    e4 = m[3] * ((h4 - h0) - (T0 + K) * (s4 - s0))
    e5 = m[4] * ((h5 - h0) - (T0 + K) * (s5 - s0))
    e6 = m[5] * ((h6 - h0) - (T0 + K) * (s6 - s0))
    e7 = m[6] * ((h7 - h0) - (T0 + K) * (s7 - s0))
    e8 = m[7] * ((h8 - h0) - (T0 + K) * (s8 - s0))
    e9 = m[8] * ((h9 - h0) - (T0 + K) * (s9 - s0))
    e10 = m[9] * ((h10 - h0) - (T0 + K) * (s10 - s0))

    if t2 > t6:
        fg_tin2 = fg_tin
        fg_tout2 = fg_calculation(fg_m, q_heater1, fg_tin2)
        fg_tin6 = fg_tout2
        fg_tout6 = fg_calculation(fg_m, q_heater2, fg_tin6)

    else:
        fg_tin6 = fg_tin
        fg_tout6 = fg_calculation(fg_m, q_heater2, fg_tin6)
        fg_tin2 = fg_tout6
        fg_tout2 = fg_calculation(fg_m, q_heater1, fg_tin2)
    hin2_fg, sin2_fg = h_s_fg(fg_tin2, P0)
    hin6_fg, sin6_fg = h_s_fg(fg_tin6, P0)
    hout2_fg, sout2_fg = h_s_fg(fg_tout2, P0)
    hout6_fg, sout6_fg = h_s_fg(fg_tout6, P0)

    e_fgin2 = fg_m * ((hin2_fg - h0_fg) - (T0 + K) * (sin2_fg - s0_fg)) + 0.5e6
    e_fgin6 = fg_m * ((hin6_fg - h0_fg) - (T0 + K) * (sin6_fg - s0_fg)) + 0.5e6
    e_fgout2 = fg_m * ((hout2_fg - h0_fg) - (T0 + K) * (sout2_fg - s0_fg)) + 0.5e6
    e_fgout6 = fg_m * ((hout6_fg - h0_fg) - (T0 + K) * (sout6_fg - s0_fg)) + 0.5e6
    # Economic Analysis
    ##Heater1
    dt1_heater = fg_tin2 - t2  # °C
    dt2_heater = fg_tout2 - t1  # °C
    UA_heater1 = (q_heater1 / 1e3) / lmtd(dt1_heater, dt2_heater)  # W / °C
    cost_heater1 = 5000 * UA_heater1  # Thesis 97/pdf116
    ##HXer
    dt1_hx = t7 - t4  # °C
    dt2_hx = t8 - t3  # °C
    UA_hx = (q_hx / 1) / lmtd(dt1_hx, dt2_hx)  # W / °C
    if t7 > 550:
        ft_hx = 1 + 0.02141 * (t7 - 550)
    else:
        ft_hx = 1
    cost_hx = 49.45 * UA_hx**0.7544 * ft_hx  # $
    ##Heater2
    dt1_heater = fg_tin6 - t6  # °C
    dt2_heater = fg_tout6 - t5  # °C
    UA_heater2 = (q_heater2 / 1e3) / lmtd(dt1_heater, dt2_heater)  # W / °C
    cost_heater2 = 5000 * UA_heater2  # Thesis 97/pdf116
    ##Turbine
    if t7 > 550:
        ft_tur = 1 + 1.137e-5 * (t7 - 550) ** 2
    else:
        ft_tur = 1
    cost_tur = 406200 * ((w_tur / 1e6) ** 0.8) * ft_tur  # $
    ##Cooler
    dt1_cooler = t9 - cw_temp  # °C
    dt2_cooler = t8 - cw_Tout(q_cooler)  # °C
    UA_cooler = (q_cooler / 1) / lmtd(dt1_cooler, dt2_cooler)  # W / °C
    if t8 > 550:
        ft_cooler = 1 + 0.02141 * (t8 - 550)
    else:
        ft_cooler = 1
    cost_cooler = 49.45 * UA_cooler**0.7544 * ft_cooler  # $
    ##Compressor
    cost_comp = 1230000 * (w_comp / 1e6) ** 0.3992  # $

    pec.append(cost_heater1)
    pec.append(cost_hx)
    pec.append(cost_heater2)
    pec.append(cost_tur)
    pec.append(cost_comp)
    pec.append(cost_cooler)
    prod_capacity = (w_tur - w_comp) / 1e6  # MW
    zk, cfuel, lcoe = economics(pec, prod_capacity)  # $/h

    c = lcoe
    thermal_efficiency = (w_tur - w_comp) / 40.53e6
    if thermal_efficiency < 0.1575:
        j = 1e5 * (0.30 - thermal_efficiency)
    else:
        j = c + 1e2 * max(0, 0.50 - q_hx / (q_heater1 + q_heater2))
    Pressure = [
        p1 / 1e5,
        p2 / 1e5,
        p3 / 1e5,
        p4 / 1e5,
        p5 / 1e5,
        p6 / 1e5,
        p7 / 1e5,
        p8 / 1e5,
        p9 / 1e5,
        p10 / 1e5,
    ]
    unit_energy = [
        q_heater1 / 1e6,
        q_hx / 1e6,
        q_heater2 / 1e6,
        w_tur / 1e6,
        q_cooler / 1e6,
        w_comp / 1e6,
    ]
    print(
        f"""
    
    Turbine Pratio = {tur_pratio:.2f}   
    Turbine output = {unit_energy[3]:.2f}MW
    Compressor Pratio = {comp_pratio:.2f} 
    Compressor Input = {unit_energy[5]:.2f}MW
    Split Ratio = {split_ratio:.2f} mass_flow = [{m[0]:.2f} {m[1]:.2f} {m[2]:.2f} {m[3]:.2f} {m[4]:.2f}]
    Temperatures =[{t1:.2f} {t2:.2f} {t3:.2f} {t4:.2f} {t5:.2f} {t6:.2f} {t7:.2f} {t8:.2f} {t9:.2f} {t10:.2f}]  Tstack={min(fg_tout2,fg_tout6):.2f}    DT ={approach_temp:.1f}
    Pressures =   [{p1/1e5:.2f} {p2/1e5:.2f} {p3/1e5:.2f} {p4/1e5:.2f} {p5/1e5:.2f} {p6/1e5:.2f} {p7/1e5:.2f} {p8/1e5:.2f} {p9/1e5:.2f} {p10/1e5:.2f}] bar
    Equipment Cost = [0.    {cost_heater1/1e3:.0f}   0.   {cost_hx/1e3:.0f}   0.   {cost_heater2/1e3:.0f}   {cost_tur/1e3:.0f}  0.   {cost_cooler/1e3:.0f}   {cost_comp/1e3:.0f}]
    Equipment Duty = Qheater=[{unit_energy[0]:.2f} {unit_energy[2]:.2f}]MW   Qcooler=[{unit_energy[4]:.2f}]MW   Qhx=[{unit_energy[1]:.1f}]MW    
    Objective Function value = {c:.2f}
    Exergy of streams = [{e1/1e6:.2f} {e2/1e6:.2f} {e3/1e6:.2f} {e4/1e6:.2f} {e5/1e6:.2f} {e6/1e6:.2f} {e7/1e6:.2f} {e8/1e6:.2f} {e9/1e6:.2f} {e10/1e6:.2f}]MW 
    Exergy of FG Streams = [{e_fgin2/1e6:.1f} {e_fgin6/1e6:.1f}]
                           [{e_fgout2/1e6:.1f} {e_fgout6/1e6:.1f}]
    Total PEC = {sum(pec):.2f} $
    Total Zk  = {sum(zk):.2f} $/h
    Heat recuperation ratio = {q_hx/(q_heater1+q_heater2):.2f}
    
    
    
        """
    )

    return c
    # Exergy costingof streams = {costs/3600*1e9} $/GJ
    # Cdiss = {cdiss:.2f} Cl = {Cl:.2f} Cp ={costs[-3]*Ep:.2f} LCOE = {lcoe:.2f} LCOEX = {lcoex:.2f}
    # Cp/Ep = {Cp/(Ep/1e6):.2f}
    # Thermal efficiency = {Ep/40.53e6*100:.2f}%


if __name__ == "__main__":
    x = [
        0.35517542824414905,
        257.12506270657195,
        11.0,
        0.0,
        414.60668503663476,
        7927205.36335623,
        32.0,
        30000000.0,
        102.15656220966437,
    ]
    result_analyses(x)
