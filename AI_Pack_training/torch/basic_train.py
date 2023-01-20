import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os

# Set pytorch download path
os.environ['TORCH_HOME'] = 'E:/Coding/AI_models/PyTorch'


# Just a format printer
def op_ed_printer(title: str, input_data, full_len: int, part):
    if part == 'op':
        print((title + ' ').ljust(full_len, '-'))

    elif part == 'ed':
        print('{}\n{}'.format(input_data, '-'.ljust(full_len, '-')))

    elif part == 'all':
        print('{}\n'.format((title + ' ').ljust(full_len, '-')),
              '{}\n{}'.format(input_data, ''.ljust(full_len, '-')))
    else:
        print('ERROR: parameter [part] not available.')


str_len = 56

# A matrix shape 5 x 3 but no initialize
tensor = torch.empty(5, 3)
op_ed_printer(title='torch.empty(5, 3)', input_data=tensor, full_len=str_len, part='all')

# Random initialized matrix
tensor = torch.rand(5, 3)
op_ed_printer(title='torch.rand(5, 3)', input_data=tensor, full_len=str_len, part='all')

# Print all zero matrix and data type = long
tensor = torch.zeros(5, 3, dtype=torch.long)
op_ed_printer(title='torch.zeros(5, 3, dtype=torch.long)', input_data=tensor, full_len=str_len, part='all')

# A tensor has 5.5 and 3 in itself
tensor = torch.tensor([5.5, 3])
op_ed_printer(title='torch.tensor([5.5, 3])', input_data=tensor, full_len=str_len, part='all')

# Create tensor base on existed tensor
op_ed_printer(title='Create tensor base on existed tensor', input_data=tensor, full_len=str_len, part='op')

print('Basic tensor x: x.new_ones(5, 3, dtype=torch.double)')
tensor = tensor.new_ones(5, 3, dtype=torch.double)
print(tensor)

print('Change tensor x use: torch.randn_like(x, dtype=torch.float)')
tensor = torch.randn_like(tensor, dtype=torch.float)

op_ed_printer(title='Create tensor base on existed tensor', input_data=tensor, full_len=str_len, part='ed')

# python data to pytorch tensor
op_ed_printer(title='Python data to Pytorch tensor', input_data=None, full_len=str_len, part='op')

data_list = [[1, 2], [3, 4]]
print('data_list: ', data_list)

tensor = torch.tensor(data_list)
op_ed_printer(title='', input_data=tensor, full_len=str_len, part='ed')

# link different tensor
op_ed_printer(title='Link different tensor by using torch.cat()', input_data=None, full_len=str_len, part='op')
tensor_1 = torch.tensor([[1, 0, 1],
                         [1, 0, 1],
                         [1, 0, 1]])
tensor_2 = torch.tensor([[1, 0, 1],
                         [1, 0, 1],
                         [1, 0, 1]])
tensor_3 = torch.tensor([[1, 0, 1],
                         [1, 0, 1],
                         [1, 0, 1]])
print('tensor_1: {}\n'
      'tensor_2: {}\n'
      'tensor_3: {}\n'.format(tensor_1, tensor_2, tensor_3))

tensor_cat_dim_1 = torch.cat([tensor_1, tensor_2, tensor_3], dim=1)
tensor_cat_dim_0 = torch.cat([tensor_1, tensor_2, tensor_3], dim=0)
tensor_cat_dim_m1 = torch.cat([tensor_1, tensor_2, tensor_3], dim=-1)
tensor_cat_dim_m2 = torch.cat([tensor_1, tensor_2, tensor_3], dim=-2)

print('Use torch.cat() to concatenate them:\n'
      '[tensor_1, tensor_2, tensor_3], dim=1:\n{}\n'
      '[tensor_1, tensor_2, tensor_3], dim=0:\n{}\n'
      '[tensor_1, tensor_2, tensor_3], dim=-1:\n{}\n'
      '[tensor_1, tensor_2, tensor_3], dim=-2:\n'.format(tensor_cat_dim_1, tensor_cat_dim_0, tensor_cat_dim_m1))

op_ed_printer(title='', input_data=tensor_cat_dim_m2, full_len=str_len, part='ed')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Use {} device'.format(device))


def data_prepare():
    # Use nn to train a model
    input_train = datasets.FashionMNIST(
        root='E:/Coding/datasets/torch',
        train=True, download=True,
        transform=ToTensor()
    )
    input_test = datasets.FashionMNIST(
        root='E:/Coding/datasets/torch',
        train=False, download=True,
        transform=ToTensor()
    )

    batch_size = 64
    train_dataloader = DataLoader(input_train, batch_size=batch_size)
    test_dataloader = DataLoader(input_test, batch_size=batch_size)

    for X, y in test_dataloader:
        print('Shape of x [N, C, H, W]: {}'.format(X.shape))
        print('Shape of y: {} {}'.format(y.shape, y.dtype))
        break

    return train_dataloader, test_dataloader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, input_model, input_loss_fn, input_optimizer):
    size = len(dataloader.dataset)
    input_model.train()

    for batch, (x_data, y_data) in enumerate(dataloader):
        x_data, y_data = x_data.to(device), y_data.to(device)

        pred_fn = input_model(x_data)
        loss = input_loss_fn(pred_fn, y_data)

        input_optimizer.zero_grad()
        loss.backward()
        input_optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x_data)
            print(f'loss: {loss:>7f} [{current:5d}/{size:>5d}]')


def test(dataloader, input_model, input_loss_fn):
    size = len(dataloader.dataset)
    print(f'Test data size: {size}')
    num_batches = len(dataloader)
    print(f'Batch: {num_batches}')

    input_model.eval()
    test_loss, correct = 0, 0

    for data_x, data_y in dataloader:
        data_x, data_y = data_x.to(device), data_y.to(device)

        pred_fn = input_model(data_x)
        test_loss += input_loss_fn(pred_fn, data_y).item()
        correct += (pred_fn.argmax(1) == data_y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test error: \n Accuracy: {(100 * correct): 0.1f}%, Avg loss: {test_loss:>8f}\n')


model = NeuralNetwork().to(device)
print(model)

train_ld, test_ld = data_prepare()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epoch = 5
for t in range(epoch):
    print(f'Epoch {t + 1}\n-------------------------------')
    train(train_ld, model, loss_fn, optimizer)
    test(test_ld, model, loss_fn)
print('Done!')

save_file = 'E:/Coding/Training/Python/AI_Pack_training/torch/sample_1.pt'
# Save model
torch.save(model.state_dict(), save_file)
print(f'Save model to: {save_file}')
# Load model
model = NeuralNetwork()
model.load_state_dict(torch.load(save_file))
print(f'Load model from: {save_file}')

# Use model for predictions
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
model.eval()

test_data = datasets.FashionMNIST(
    root='E:/Coding/datasets/torch',
    train=False, download=True,
    transform=ToTensor()
)

test_x, test_y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(test_x)
    predicted, actual = classes[pred[0].argmax(0)], classes[test_y]
    print(f'Predicted: {predicted}, Actual: {actual}')


