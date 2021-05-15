import random

import numpy

list_contain = numpy.array([], dtype=bool)


def probability_change(probability, input_list):
    changes = 100 * probability
    changes = int(round(changes, 0))

    for index in range(0, changes):

        pick = random.randint(0, numpy.size(input_list))

        if input_list[pick]:
            input_list[pick] = False

        else:
            index -= 1

    return input_list


# ==============================================================

random.seed(10)

for i in range(10):
    a = random.randint(0, 3)
    list_contain = numpy.append(list_contain, a)

list_contain = numpy.append(list_contain, 4)
list_contain = numpy.append(list_contain, 5)

for i in range(6, 0, -1):
    list_contain = numpy.append(list_contain, i)

# ---------------------------------------------

list_probability = numpy.array([])

for i in range(0, 100):
    list_probability = numpy.append(list_probability, True)

list_test = probability_change(0.2, list_probability)
