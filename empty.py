from pyfluids import Fluid, FluidsList, Input, Mixture

a = Fluid(FluidsList.CarbonDioxide).with_state(
    Input.temperature(100), Input.pressure(101.3e5)
)
b = Fluid(FluidsList.CarbonDioxide).with_state(
    Input.temperature(200), Input.pressure(101.3e5)
)
c = Fluid(FluidsList.CarbonDioxide).mixing(1, a, 2, b)
print(c.temperature)
