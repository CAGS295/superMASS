import numpy as np
from superMASS import mass

a = np.array([0.0, 1.0, 2., 3., 5., 6.])
b = np.array([2.0, 3.0])
c = mass.batched(a, b, 1, 6, 1)
print(c)
