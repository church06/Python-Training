import numpy

import SparseLinearModel

x = numpy.array([range(0, 1000)])
voxels = 1000

y = SparseLinearModel.linear_regression(x, voxels)
print('y: ', y.shape)

