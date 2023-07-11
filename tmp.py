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
