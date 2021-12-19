import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

trans = transforms.Compose([transforms.Resize((28, 28)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])

train_data = torchvision.datasets.ImageFolder(root=r'./img/train', transform=trans)
val_data = torchvision.datasets.ImageFolder(root=r'./img/val', transform=trans)
test_data = torchvision.datasets.ImageFolder(root=r'./img/test', transform=trans)
# print(train_data.shape)
batch_size = 64
train_data_loader = data.DataLoader(train_data, batch_size)
val_data_loader = data.DataLoader(val_data, batch_size)
test_data_loader = data.DataLoader(test_data, batch_size)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # создаем 3 полносвязных слоя и
        # указываем колличество входных и выходных нейронов
        self.fc1 = nn.Linear(28 * 28 * 3, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28 * 3)
        # указываем функцию активации на каждом слое
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
