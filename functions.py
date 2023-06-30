from pyfluids import Fluid, FluidsList, Input
import numpy as np


##Specific heat calculation works fine with DT similar to estimate h2-h1
##However, h2 =/= cp*T_hotout
def lmtd(dthin, dt2):
    return (dthin - dt2) / np.log(dthin / dt2)


def enthalpy_entropy(T, P):
    """
    Takes the the temperature and pressure of a CO2 stream and gives enthalpy, entropy and specific heat values at that temperature
    Temperature input is C, Pressure input is pa
    Return: Enthalpy (J/kg), Entropy (J/kgK), Specific Heat (J/kgK)

    """
    substance = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    return (substance.enthalpy, substance.entropy)


def specific_heat(T, P):
    substance = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    return substance.specific_heat


T0 = 15
P0 = 101325
K = 273.15
(h0, s0) = enthalpy_entropy(T0, P0)


def temperature(h, P):
    """
    Takes the the temperature and pressure of a CO2 stream and gives enthalpy, entropy and specific heat values at that temperature
    Temperature input is C, Pressure input is pa
    Return: Enthalpy (J/kg), Entropy (J/kgK), Specific Heat (J/kgK)

    """
    substance = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.enthalpy(h), Input.pressure(P)
    )
    return substance.temperature


def pinch_calculation(T_hin, H_hotin, T_coldin, H_coldin, P_hotout, P_coldout, m):
    list_T_hotout = [T_hotout for T_hotout in range(int(T_coldin) + 5, int(T_hin))]
    if len(list_T_hotout) == 0:
        return (0, 0)
    h2 = list()
    for temp in list_T_hotout:
        a, _ = enthalpy_entropy(temp, P_hotout)
        h2.append(a)
    h2 = np.asarray(h2)
    q_hx1 = m * H_hotin - m * h2
    list_T_coldout = [T_coldout for T_coldout in range(int(T_coldin), int(T_hin) - 5)]
    if len(list_T_coldout) == 0:
        return (0, 0)
    h5 = list()
    for temp in list_T_coldout:
        a, _ = enthalpy_entropy(temp, P_coldout)
        h5.append(a)
    h5 = np.asarray(h5)
    q_hx2 = m * h5 - m * H_coldin
    q_hx = q_hx1 - q_hx2
    index = np.where(q_hx[:-1] * q_hx[1:] < 0)[0]
    if len(index) == 0:
        return (0, 0)
    T_hotout = list_T_hotout[index[0]]
    T_coldout = list_T_coldout[index[0]]
    return (T_hotout, T_coldout)


def Pressure_calculation(tur_pratio, comp_pratio):
    # [p1,p2,p3,p4,p5,p6]
    pres = np.array(
        [
            [1, 0, 0, 0, 0, -1 / tur_pratio],
            [1, -1, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
            [0, 0, comp_pratio, -1, 0, 0],
            [0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, -1],
        ]
    )
    dp = np.array([0, 1e5, 0.5e5, 0, 1e5, 1e5]).reshape(-1, 1)
    try:
        pressures = np.linalg.solve(pres, dp)
    except:
        # print("singular matrix", tur_pratio, comp_pratio)
        return [0, 0, 0, 0, 0, 0]
    p1 = pressures.item(0)
    if p1 < 0:
        # print("negative Pressure")
        return [0, 0, 0, 0, 0, 0]
    ub = 300e5 / max(pressures)
    lb = 74e5 / max(pressures)
    pres_coeff = np.random.uniform(lb, ub)
    pressures = pres_coeff * pressures
    p1 = pressures.item(0)
    p2 = pressures.item(1)
    p3 = pressures.item(2)
    p4 = pressures.item(3)
    p5 = pressures.item(4)
    p6 = pressures.item(5)
    return (p1, p2, p3, p4, p5, p6)
