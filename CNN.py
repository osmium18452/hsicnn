import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, bands, patch_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(bands, 300, 3, 1, 1),  # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(kernel_size=2,
                         stride=1,
                         padding=0),
            nn.ReLU(),
            nn.Conv2d(300, 200, 3, 1, 1),  # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(kernel_size=2,
                         stride=1,
                         padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(200*(patch_size-2)*(patch_size-2),200),
            nn.ReLU(),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.Linear(100,num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.conv1(x)
        return out

if __name__ == '__main__':
    cnn=CNN(100,7,16)
    input=torch.zeros((1,100,7,7))
    print(cnn(input).shape)