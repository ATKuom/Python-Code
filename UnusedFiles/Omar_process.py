import numpy as np
from scipy.optimize import fsolve

# Reactor parameters (units: kmol/m³, kJ/kmol, K)
V = 3
F = 12.129827011009338
C_A0 = 2.34793126
C_B0 = 45.1432654
C_C0 = 0.0206103519
C_D0 = 0.0
T0 = 355.0


# Stoichiometry matrix [A, B, C, D]
nu = np.array([[-1, -1, 1, 0], [-1, 0, -1, 1]])

# Enthalpy data [h0, slope] for A, B, C, D
hl_data = np.array(
    [
        [-143.8, 0.2037],  # A
        [-320.0, 0.1068],  # B
        [-519.0, 0.1924],  # C
        [-723.0, 0.3043],  # D
    ]
)


def H(T):
    return hl_data[:, 0] + hl_data[:, 1] * T


def rate(C, T):
    C_A, C_B, C_C, C_D = C
    k1 = 13706.91 * np.exp(-8220 / T)
    r1 = k1 * C_A * C_B
    k2 = 96341.59 * np.exp(-8700 / T)
    r2 = k2 * C_A * C_C
    return r1, r2


def equations(vars):
    C_A, C_B, C_C, C_D, T = vars
    r1, r2 = rate([C_A, C_B, C_C, C_D], T)
    delta_H_rxn1 = -0.1181 * T - 55.2
    delta_H_rxn2 = -0.0918 * T - 60.2

    resid_A = C_A - (C_A0 + (V / F) * (nu[0, 0] * r1 + nu[1, 0] * r2))
    resid_B = C_B - (C_B0 + (V / F) * (nu[0, 1] * r1 + nu[1, 1] * r2))
    resid_C = C_C - (C_C0 + (V / F) * (nu[0, 2] * r1 + nu[1, 2] * r2))
    resid_D = C_D - (C_D0 + (V / F) * (nu[0, 3] * r1 + nu[1, 3] * r2))

    h_A0, h_B0, h_C0, h_D0 = H(T0)
    h_A, h_B, h_C, h_D = H(T)
    delta_h = (
        C_A * (h_A - h_A0)
        + C_B * (h_B - h_B0)
        + C_C * (h_C - h_C0)
        + C_D * (h_D - h_D0)
    )
    Q_rxn = (delta_H_rxn1 * r1 + delta_H_rxn2 * r2) * V
    resid_T = (delta_h * F) - Q_rxn

    return [resid_A, resid_B, resid_C, resid_D, resid_T]


# Adjusted initial guess
guess = [C_A0, C_B0, C_C0, C_D0, T0]
solution = fsolve(equations, guess)

# Print results
C_A_sol, C_B_sol, C_C_sol, C_D_sol, T_sol = solution
print("Adiabatic CSTR Results:")
print(f"C_A: {C_A_sol:.3f} kmol/m³")
print(f"C_B: {C_B_sol:.3f} kmol/m³")
print(f"C_C: {C_C_sol:.3f} kmol/m³")
print(f"C_D: {C_D_sol:.3f} kmol/m³")
