import random

import matplotlib.pyplot as plt
list = []

random.seed(10)

for i in range(50):
    a = random.randint(0, 5)
    list.append(a)

plt.figure(figsize=(5, 2))

plt.plot(list)

plt.show()