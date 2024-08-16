import sys

sys.path.append("..")
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallSoulaweenNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(SoulaweenNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, padding=1)
        self.conv4= nn.Conv2d(args.num_channels, args.num_channels, 3)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels * (self.board_x-2) * (self.board_y-2), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.args.num_channels * (self.board_x-2) * (self.board_y-2))

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))),
            p=self.args.dropout,
            training=self.training,
        )
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))),
            p=self.args.dropout,
            training=self.training,
        )

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

class SoulaweenNNet(nn.Module):
    def __init__(self, game, args):
        super(SoulaweenNNet, self).__init__()
        
        # Game parameters
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)  # 4x4 -> 4x4
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)  # 4x4 -> 4x4

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)

        # Fully connected layers
        self.fc1 = nn.Linear(args.num_channels * 4 * 4, 128)  # 4x4 -> Flatten to 16*num_channels
        self.fc_bn1 = nn.BatchNorm1d(128)

        # Policy head
        self.fc_policy = nn.Linear(128, self.action_size)

        # Value head
        self.fc_value = nn.Linear(128, 1)

    def forward(self, s):
        # Reshape the input to fit the convolutional layers
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x 4 x 4
        
        # Convolutional layers with ReLU and Batch Normalization
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x 4 x 4
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x 4 x 4

        # Flatten for the fully connected layer
        s = s.view(-1, self.args.num_channels * 4 * 4)  # batch_size x (num_channels * 16)

        # Fully connected layer with dropout
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 128

        # Policy head: produces a log-softmax distribution over actions
        pi = self.fc_policy(s)  # batch_size x action_size
        pi = F.log_softmax(pi, dim=1)

        # Value head: produces a scalar value
        v = self.fc_value(s)  # batch_size x 1
        v = torch.tanh(v)

        return pi, v
