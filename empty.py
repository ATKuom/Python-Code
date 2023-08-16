import numpy as np
from split_functions import fg_calculation

fg_m = 68.75

heater_position = [0, 5, 6]
q_heater = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 5466954.45221552, 11421174.90100743])
Temperatures = np.array(
    [
        431.3698234,
        59.08819002,
        32.95478462,
        48.20620119,
        400.46996853,
        503.48001517,
        504.05666833,
    ]
)
print(q_heater / 1e6)
descending_Temp = np.sort(Temperatures[heater_position])[::-1]
fg_tin = 539.76
for Temp in descending_Temp:
    index = np.where(Temperatures == Temp)[0][0]
    fg_tout = fg_calculation(fg_m, q_heater[index], fg_tin)
    dt1_heater = fg_tin - Temperatures[index]
    dt2_heater = fg_tout - Temperatures[index - 1]
    fg_tin = fg_tout
