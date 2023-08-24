import numpy as np
from pyfluids import Fluid, Input, FluidsList
from split_functions import lmtd

lmtd(3, 3)
# Thotin = 245.39
# photin = 79e5
# tcoldin = 195.25
# pcoldin = 233e5
# hx_pdrop = 0.5e5
# m_hotside = 56.33
# m_coldside = 19.85
# hhotin = 466367
# hcoldin = 637797
# dt = 9.4
# hotside_outlet = (
#     Fluid(FluidsList.CarbonDioxide)
#     .with_state(Input.temperature(Thotin), Input.pressure(photin))
#     .cooling_to_temperature(tcoldin + dt, hx_pdrop)
# )
# breakpoint()
# dh_hotside = hhotin - hotside_outlet.enthalpy
# q_hotside = dh_hotside * m_hotside
# dh_coldside = q_hotside / m_coldside
# print(dh_hotside, q_hotside / 1e6, dh_coldside)
# coldside_outlet = (
#     Fluid(FluidsList.CarbonDioxide)
#     .with_state(Input.temperature(tcoldin), Input.pressure(pcoldin))
#     .heating_to_enthalpy(hcoldin + dh_coldside, hx_pdrop)
# )
# q_hx = q_hotside

# coldside_outlet = (
#     Fluid(FluidsList.CarbonDioxide)
#     .with_state(Input.temperature(tcoldin), Input.pressure(pcoldin))
#     .heating_to_temperature(Thotin - dt, hx_pdrop)
# )
# dh_coldside = coldside_outlet.enthalpy - hcoldin
# q_coldside = dh_coldside * m_coldside
# dh_hotside = q_coldside / m_hotside
# hotside_outlet = (
#     Fluid(FluidsList.CarbonDioxide)
#     .with_state(Input.temperature(Thotin), Input.pressure(photin))
#     .cooling_to_enthalpy(hhotin - dh_hotside, hx_pdrop)
# )
# q_hx = q_coldside
