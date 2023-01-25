import os
import torch
import numpy

os.environ['TORCH_HOME'] = 'E:/Coding/AI_models/PyTorch'

data = [[1, 2], [3, 4], [5, 6]]
data_x = torch.tensor(data)
print(f'Python Lists -> torch.tensor():\n{data_x}\n')

np_array = numpy.array(data)
data_x_np = torch.from_numpy(np_array)
print(f'Numpy array -> torch.tensor():\n{data_x_np}\n')

# Pytorch ones & rand tensor
data_ones = torch.ones_like(data_x)
data_rand = torch.rand_like(data_x)
print(f'Ones Tensor:\n{data_ones}\n')
print(f'Rand Tensor:\n{data_rand}\n')

# Create a tensor with a given shape
shape = (2, 3)

tensor_rand = torch.rand(shape)
tensor_ones = torch.ones(shape)
tensor_zeros = torch.zeros(shape)

print(f'Create tensors with a given shape:\n'
      f'shape: {shape}\n'
      f'torch.rand():\n{tensor_rand}\n'
      f'torch.ones():\n{tensor_ones}\n'
      f'torch.zeros():\n{tensor_zeros}')


