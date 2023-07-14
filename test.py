from pyfluids import Fluid, FluidsList, Input

heater = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(206.1), Input.pressure(238.9e5))
    .heating_to_temperature(411.4, 0)
)
h6, s6, t6, p6 = heater.enthalpy, heater.entropy, heater.temperature, heater.pressure
turb = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(t6), Input.pressure(p6))
    .expansion_to_pressure(78.5e5, 85)
)
h1, s1, t1, p1 = turb.enthalpy, turb.entropy, turb.temperature, turb.pressure

hxer_hotside = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(t1), Input.pressure(p1))
    .cooling_to_temperature(81.4, 0.7e5)
)
h2, s2, t2, p2 = (
    hxer_hotside.enthalpy,
    hxer_hotside.entropy,
    hxer_hotside.temperature,
    hxer_hotside.pressure,
)
cooler = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(t2), Input.pressure(p2))
    .cooling_to_temperature(32.3, 0.8e5)
)
h3, s3, t3, p3 = cooler.enthalpy, cooler.entropy, cooler.temperature, cooler.pressure
comp = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(t3), Input.pressure(p3))
    .compression_to_pressure(241.3e5, 82)
)
h4, s4, t4, p4 = comp.enthalpy, comp.entropy, comp.temperature, comp.pressure
hxer_coldside = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(t4), Input.pressure(p4))
    .heating_to_temperature(206.1, 2.4e5)
)
h5, s5, t5, p5 = (
    hxer_coldside.enthalpy,
    hxer_coldside.entropy,
    hxer_coldside.temperature,
    hxer_coldside.pressure,
)
delta_t = 10
t2 = t4 + delta_t
import scipy.optimize as opt


def objective(t):
    print(t)
    hotside = (
        Fluid(FluidsList.CarbonDioxide)
        .with_state(Input.temperature(t1), Input.pressure(p1))
        .cooling_to_temperature(t[0], 1e5)
    )
    coldside = (
        Fluid(FluidsList.CarbonDioxide)
        .with_state(Input.temperature(t4), Input.pressure(p5))
        .heating_to_temperature(t[1], 1e5)
    )
    return hotside.enthalpy - coldside.enthalpy


x0 = [t1 - delta_t, t4 + delta_t]
t2, t5 = opt.brent(objective, x0)
print(t2)
