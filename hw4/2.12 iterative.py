import numpy as np
import math
def iterative(n):
    return (8 / (0.05**2) * math.log((4*((2*n)**10 + 1))/0.05)
)
n = 10
for _ in range(1000):
    n = iterative(n)
print(n)