import h5py
import numpy

print("Loading data...")

data = numpy.array([])

with h5py.File('G:/Entrance/Coding_Training/Python Program/MSc project/GenericObjectDecoding/data/Subject1.h5',
               'r') as h5f:
    data = numpy.array(h5f)

print(data)
