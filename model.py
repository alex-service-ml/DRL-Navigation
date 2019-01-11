import torch
import torch.nn as nn
import torch.nn.functional as F


class BananaNet(nn.Module):

    def __init__(self, state_size, action_size):
        super(BananaNet, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class VisualBananaNet(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        print(state_size)
        print('ASSUMING 84x84x4 INPUT')
        self.conv1 = nn.Conv2d(state_size[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)  # TODO: Hard-coded based on DQN paper
        self.fc2 = nn.Linear(512, action_size)
        pass

    def forward(self, frame_stack):
        x = F.relu(self.conv1(frame_stack))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.reshape(x.size(0), -1)))  # TODO: Verify reshape function
        x = self.fc2(x)
        return x


class BananaResNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(BananaResNet, self).__init__()

        self.blk1fc1 = nn.Linear(state_size, 128)
        self.blk1fc2 = nn.Linear(128, 128)
        self.blk1fc3 = nn.Linear(128, 64)

        self.blk2fc1 = nn.Linear(64, 64)
        self.blk2fc2 = nn.Linear(64, 64)
        self.blk2fc3 = nn.Linear(64, 32)

        self.outfc = nn.Linear(32, action_size)

    def forward(self, state):
        skip = F.relu(self.blk1fc1(state))
        x = F.relu(self.blk1fc2(skip))
        x = x + skip
        skip = F.relu(self.blk1fc3(x))
        x = F.relu(self.blk2fc1(skip))
        skip = x + skip
        x = F.relu(self.blk2fc2(x))
        skip = x + skip
        x = F.relu(self.blk2fc3(skip))
        x = self.outfc(x)

        return x


class PERLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target, weights):
        ret = (input - target) ** 2
        ret = weights * ret
        ret = torch.mean(ret)
        return ret
