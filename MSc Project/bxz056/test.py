import numpy as np

import SparseLinearModel

x = np.array([range(0, 1000)])
voxels = 1000

y = SparseLinearModel.linear_regression(x, voxels)
print('y: ', y.shape)

# -------------------------------------------------------------------------------------------

test = np.ones((2, 3))
print('test: ', test.shape)

test = test.T
print('test after T: ', test.shape)

# -------------------------------------------------------------------------------------------
axis = 1
data = np.array([(1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6, 7)])
value = np.array([(1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6, 7)])
num = 500

num_elem = data.shape[axis]

sorted_index = np.argsort(value)[::-1]

rank = np.zeros(num_elem, dtype=int)
rank[sorted_index] = np.array(range(0, num_elem))

selected_index_bool = rank < num

print('num: ', num)
print('rank: ', rank)
print('selected_index_bool: ', selected_index_bool)

if axis == 0:
    selected_data = data[selected_index_bool, :]
    selected_index = np.array(range(0, num_elem), dtype=int)[selected_index_bool]
elif axis == 1:
    selected_data = data[:, selected_index_bool]
    selected_index = np.array(range(0, num_elem), dtype=int)[selected_index_bool]
else:
    selected_index = np.array([])

print(selected_index.shape)
