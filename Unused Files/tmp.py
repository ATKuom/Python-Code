# import sys

# MY_COOL_RESULT = None


# def main(arg):
#     print(arg)
#     first_run = True

#     def f(x):
#         return x + first_run

#     global MY_COOL_RESULT
#     MY_COOL_RESULT = 1


# main(sys.argv[0])
# print(sys.argv)

# 1. Edit config.py locally (and ignore changes)
# 2. Create default data path where you control it (but still ignore that)
# 3. Use commandline argument / environment variable
# workflow manager
# class ObjectiveFunctionObject:
#     first_run = True

#     def f(self, x):
#         ...
#         if self.first_run:
#             ...
#             self.first_run = False
#         else:
#             ...
#         ...
#     def __call__(self, ...)

# evalute(obj.f)
# obj = ObjectiveFunctionObject()
# y = objective_function(x)
# objective_function + 5

# FIRST_RUN=True
# if FIRST_RUN:
#         t6 = ???
#         FIRST_RUN = False
#     else:


# class ConstraintsViolatedError(ValueError):
#     pass

# def function(...):
#     y = ...
#     if constraints_are_violated(y):
#         raise ConstraintsViolatedError
#     ...
#     return ...

# if constraints_are_met(x):
#     try:
#         result = scipy.optimize.minimize(function, x0, )
#     except ConstraintsViolatedError:
#         return PENALTY_VALUE
# else:
#     return PENALTY_VALUE

# 0. pec = [cost_tur, ...]
# 1. pec += [cost_tur, ...]
# 2. pec.extend([cost_tur, ...])
# 3.

# cost list creation rather than array
# m1 = [
#     i[:]
#     for i in [[0] * (len(equipment) + equipment.count(1) + equipment.count(3) + 3)]
#     * len(equipment)
# ]

# UNIT_FROM_CHAR = {'T': Turbine}

# def word_to_units(sequence):
#     return [UNIT_FROM_CHAR[char]() for char in sequence]


# if __name__ == "__main__":
#     layout = expert_designs
#     word_to_units(layout[1])

# [T,A,C,H,T,T,C]
# class for units?
# while look without backup or some kind of alternative ending to stop
# #####Profiling if it is slow? Finding the real cause of the speed problems not just assumptions, programs or timing each part to see the performance
# data oriented programming

# class Turbine:
#     DecisionVariables = namedtuple('DecisionVariables', 'Pressures[i]')
#     var: DecisionVariables

#     def __init__(self, *var):
#         if len(var) == 0:
#             self.var = #assign randomly
#         self.var = DecisionVariables(var)

#     def outlet_parameters(self, *inlet_parameters):
#         # do computation based on var
#         return outlet_parameters
