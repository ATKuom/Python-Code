# import pandas as pd
# import numpy as np
# import config

# np.printoptions(threshold=np.inf)
# pd.set_option("display.max_rows", None)
# datalist = np.load(config.DATA_DIRECTORY / "v4D0_m1.npy", allow_pickle=True)
# datalist2 = np.load(config.DATA_DIRECTORY / "v4D1_m1.npy", allow_pickle=True)
# index = np.where(np.isin(datalist2, datalist, invert=True))[0]
# new_ones = datalist2[index]
# dataset = pd.DataFrame(new_ones)
# print(dataset)

"""
To execute scripts, you neeed to use eng.script_name(nargout=0)
To execute functions, you need to use eng.function_name(args), if it does not require args then eng.function_name(), if you put nargout=0, it will return None
I can run the python functions in matlab using py.filename.function_name(args)
"""
# import matlab.engine
# from pyfluids import Fluid, FluidsList, Input


# def ptrial(x):
#     return 100 * (x[1] - x[0] ^ 2) ^ 2 + (1 - x[0]) ^ 2


# eng = matlab.engine.start_matlab()
# x0 = matlab.double([-1.0, 2.0])
# A = matlab.double([1.0, 2.0])
# b = matlab.double([1.0])
# x = eng.fmincon(ptrial(), x0, A, b)
# # x = eng.x1(nargout=0)
# # x = eng.trial()
# print(x)


# def turbine(tin, pin, pout):
#     ntur = 85
#     m = 100
#     turb_out = (
#         Fluid(FluidsList.CarbonDioxide)
#         .with_state(Input.temperature(tin), Input.pressure(pin))
#         .expansion_to_pressure(pout, ntur)
#     )
#     turb_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
#         Input.temperature(tin), Input.pressure(pin)
#     )
#     delta_h = turb_inlet.enthalpy - turb_out.enthalpy
#     w_tur = delta_h * m
#     return w_tur


# a = turbine(300, 10, 5)
