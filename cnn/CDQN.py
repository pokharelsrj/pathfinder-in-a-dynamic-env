from torch import nn
import torch.nn.functional as F


class CDQN(nn.Module):
    """CNN-Based Deep Q-Network Model for 8×8 grid inputs."""

    def __init__(self):
        super(CDQN, self).__init__()
        # conv layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
