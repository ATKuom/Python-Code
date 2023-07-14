import numpy as np
from designs import expert_designs


# The output of ML model will be one-hot encoded strings of the layouts
# The output will be cut down using start and end tokens to identify the units in the layout
# Known things from the string: Number of units, their placements, and their connections
# Decision variables are added to the PSO based on their units and placements
# Unit input T and P of i unit = T(i-1), P(i-1) Unit output T and P of i unit = T(i), P(i)
# Pressures of the system are identified then calculated using turbine and compressor pressure ratios
# Temperatures of the system are identified then inputted using Cooler and heater target temperatures
# Mass flow of streams are identified then calculated using mass variable and splitter ratios
# Order of calculations can be done by checking if T(i-1) and P(i-1) are known, if so, calculate the unit function and get T(i) and P(i)

# In my mind there is two different versions of implementation, one is a while loop which constantly checks to obtain all values for T and P and when it is able to obtain them
# It can progress further with exergy analysis,economy analysis and exergoeconomic analysis to obtain the objective function value for the layout and the variables
# It is implemented like checking to see if the input values are known for the functions, if they are, calculate the function and get the output values
# The other version is finding a way to order the functions in a way that the input values are known for the functions, if they are, calculate the function and get the output values

# UNIT_FROM_CHAR = {'T': Turbine}

# def word_to_units(sequence):
#     return [UNIT_FROM_CHAR[char]() for char in sequence]


# if __name__ == "__main__":
#     datalist = expert_designs
#     word_to_units(datalist[1])

# [T,A,C,H,T,T,C]
# class for units?
# while look without backup or some kind of alternative ending to stop
# #####Profiling if it is slow? Finding the real cause of the speed problems not just assumptions, programs or timing each part to see the performance
# data oriented programming

# class Turbine:
#     DecisionVariables = namedtuple('DecisionVariables', 'outlet_pressure')
#     var: DecisionVariables

#     def __init__(self, *var):
#         if len(var) == 0:
#             self.var = #assign randomly
#         self.var = DecisionVariables(var)

#     def outlet_parameters(self, *inlet_parameters):
#         # do computation based on var
#         return outlet_parameters
