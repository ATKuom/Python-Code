import numpy as np

enumerated_equipment = [
    (0, 9),
    (1, 4),
    (2, 7),
    (3, 5),
    (4, 7),
    (5, 4),
    (6, 1),
    (7, 5),
    (8, 2),
    (9, 3),
]
Pressures = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10, 0.0, 0.0, 15])
equipment = [9, 4, 7, 5, 7, 4, 1, 5, 2, 3]
cooler_pdrop = 1
heater_pdrop = 0
hx_pdrop = 0.5
equipment = np.asarray(equipment)
if np.where(7 == equipment)[0].size != 0:
    mixer1, mixer2 = np.where(7 == equipment)[0]


for i in range(3):
    if Pressures[mixer1 - 1] != 0 and Pressures[mixer2 - 1] != 0:
        Pressures[mixer2] = min(Pressures[mixer1 - 1], Pressures[mixer2 - 1])
    for i in range(len(Pressures)):
        if Pressures[i] != 0:
            if i == len(Pressures) - 1:
                if equipment[0] == 2:
                    Pressures[0] = Pressures[i] - cooler_pdrop
                if equipment[0] == 4:
                    Pressures[0] = Pressures[i] - heater_pdrop
                if equipment[0] == 5:
                    Pressures[0] = Pressures[i] - hx_pdrop
                if equipment[0] == 9:
                    Pressures[0] = Pressures[i]
                    Pressures[mixer1] = Pressures[i]

            else:
                if equipment[i + 1] == 2:
                    Pressures[i + 1] = Pressures[i] - cooler_pdrop
                if equipment[i + 1] == 4:
                    Pressures[i + 1] = Pressures[i] - heater_pdrop
                if equipment[i + 1] == 5:
                    Pressures[i + 1] = Pressures[i] - hx_pdrop
                if equipment[i + 1] == 9:
                    Pressures[i + 1] = Pressures[i]
                    Pressures[mixer1] = Pressures[i]
    print(Pressures)
