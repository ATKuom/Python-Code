import random
import matplotlib.pyplot as plt
from exergy_calculation import enthalpy_entropy, h0, s0, T0, K


# ------------------------------------------------------------------------------
def objective_function(O):
    t6 = O[0]
    t1 = O[1]
    tur_pratio = O[2]
    m = O[3]
    p6 = O[4]
    p1 = O[5]
    # wt = O[6]
    nt = 0.93
    gamma = 1.28

    t1 = (t6 + K) - nt * ((t6 + K) - (t6 + K) / (tur_pratio ** (1 - 1 / gamma))) - K
    p1 = p6 / tur_pratio
    (hin, sin, _) = enthalpy_entropy(t6, p6)
    (hout, sout, _) = enthalpy_entropy(t1, p1)
    ein_tur = m * (hin - h0 - T0 * (sin - s0))
    eout_tur = m * (hout - h0 - T0 * (sout - s0))
    wt = max(m * (hin - hout), 0.1)
    fuel_tur = ein_tur - eout_tur
    if t6 > 550:
        ft = 1 + 1.106e-4 * (t6 - 550) ** 2
    else:
        ft = 1
    cost_tur = 182600 * (wt**0.5561) * ft
    cost_prod_execo_tur = (fuel_tur + cost_tur) / wt
    z = cost_prod_execo_tur
    # print(t6, p6 / 1e6, t1, p1 / 1e6, wt / 1e6)

    return z


bounds = [
    (250, 560),
    (35, 560),
    (1, 25),
    (0.1, 100),
    (74e6, 250e6),
    (74e6, 250e6),
    # (0.1, 1e15),
]  # upper and lower bounds of variables
nv = len(bounds)  # number of variables
mm = -1  # if minimization mm, mm = -1; if maximization mm, mm = 1

# PARAMETERS OF PSO
particle_size = 7  # number of particles
iterations = 30  # max number of iterations
w = 0.8  # inertia constant
c1 = 1  # cognative constant
c2 = 2  # social constant

# Visualization
fig = plt.figure()
ax = fig.add_subplot()
fig.show()
plt.title("Evolutionary process of the objective function value")
plt.xlabel("Iteration")
plt.ylabel("Objective function")


# ------------------------------------------------------------------------------
class Particle:
    def __init__(self, bounds):
        self.particle_position = []
        self.particle_velocity = []
        self.local_best_particle_position = []
        self.fitness_local_best_particle_position = (
            initial_fitness  # objective function value of the best particle position
        )
        self.fitness_particle_position = (
            initial_fitness  # objective function value of the particle position
        )

        for i in range(nv):
            self.particle_position.append(
                random.uniform(bounds[i][0], bounds[i][1])
            )  # generate random initial position
            self.particle_velocity.append(
                random.uniform(-1, 1)
            )  # generate random initial velocity

    def evaluate(self, objective_function):
        self.fitness_particle_position = objective_function(self.particle_position)
        if mm == -1:
            if (
                self.fitness_particle_position
                < self.fitness_local_best_particle_position
            ):
                self.local_best_particle_position = (
                    self.particle_position
                )  # update particle's local best poition
                self.fitness_local_best_particle_position = (
                    self.fitness_particle_position
                )  # update fitness at particle's local best position
        if mm == 1:
            if (
                self.fitness_particle_position
                > self.fitness_local_best_particle_position
            ):
                self.local_best_particle_position = (
                    self.particle_position
                )  # update particle's local best position
                self.fitness_local_best_particle_position = (
                    self.fitness_particle_position
                )  # update fitness at particle's local best position

    def update_velocity(self, global_best_particle_position):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()

            # local explorative position displacement component
            cognitive_velocity = (
                c1
                * r1
                * (self.local_best_particle_position[i] - self.particle_position[i])
            )

            # position displacement component towards global best
            social_velocity = (
                c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            )

            self.particle_velocity[i] = (
                w * self.particle_velocity[i] + cognitive_velocity + social_velocity
            )

    def update_position(self, bounds):
        for i in range(nv):
            self.particle_position[i] = (
                self.particle_position[i] + self.particle_velocity[i]
            )

            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][1]
            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][0]


class PSO:
    def __init__(self, objective_function, bounds, particle_size, iterations):
        fitness_global_best_particle_position = initial_fitness
        global_best_particle_position = []
        swarm_particle = []
        for i in range(particle_size):
            swarm_particle.append(Particle(bounds))
        A = []

        for i in range(iterations):
            for j in range(particle_size):
                swarm_particle[j].evaluate(objective_function)

                if mm == -1:
                    if (
                        swarm_particle[j].fitness_particle_position
                        < fitness_global_best_particle_position
                    ):
                        global_best_particle_position = list(
                            swarm_particle[j].particle_position
                        )
                        fitness_global_best_particle_position = float(
                            swarm_particle[j].fitness_particle_position
                        )
                if mm == 1:
                    if (
                        swarm_particle[j].fitness_particle_position
                        > fitness_global_best_particle_position
                    ):
                        global_best_particle_position = list(
                            swarm_particle[j].particle_position
                        )
                        fitness_global_best_particle_position = float(
                            swarm_particle[j].fitness_particle_position
                        )

            for j in range(particle_size):
                swarm_particle[j].update_velocity(global_best_particle_position)
                swarm_particle[j].update_position(bounds)

            A.append(fitness_global_best_particle_position)  # record the best fitness

            # Visualization
            # ax.plot(A, color="r")
            # fig.canvas.draw()
            # ax.set_xlim(left=max(0, i - iterations), right=i + 3)
        print("Result:")
        print("Optimal solutions:", global_best_particle_position)
        print("Objective function value:", fitness_global_best_particle_position)
        plt.plot(A)


# ------------------------------------------------------------------------------
if mm == -1:
    initial_fitness = float("inf")  # for minimization problem

if mm == 1:
    initial_fitness = -float("inf")  # for maximization problem

# ------------------------------------------------------------------------------
# Main PSO
PSO(objective_function, bounds, particle_size, iterations)
plt.show()
