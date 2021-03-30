pi = 0
for k in range(10000):
    pi += 1 / pow(16, k) * (4 / (8 * k + 1) - 2 / (8*k + 4) - 1 / (8 * k + 5) - 1 / (8 * k + 6))
print('π1 = ', pi)

from random import random
from time import perf_counter
darts = 10000 * 1000
hits = 0.0
start = perf_counter()
for i in range(1, darts + 1):
    x, y = random(), random()
    dist = pow(x ** 2 + y ** 2, 0.5)
    if  dist <= 1.0:
        hits += 1
pi2 = 4 * (hits / darts)
print('π2 = {}'.format(pi2))
print('run time: {:.5f}'.format(perf_counter() - start))