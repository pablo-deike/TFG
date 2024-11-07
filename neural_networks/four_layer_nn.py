import torch.nn as nn


class FourLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FourLayerNN, self).__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, 750)
        self.fc3 = nn.Linear(750, 600)
        self.fc4 = nn.Linear(600, 800)
        self.fc5 = nn.Linear(800, output_dim)
        self.internal = nn.ReLU()
        self.activation = nn.ReLU()
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.batchnorm1 = nn.BatchNorm1d(
            1000, eps=1e-05, momentum=None
        )  # batch norm should be frozen when evaluating the model
        self.batchnorm2 = nn.BatchNorm1d(750, eps=1e-05, momentum=None)
        self.batchnorm3 = nn.BatchNorm1d(600, eps=1e-05, momentum=None)
        self.batchnorm4 = nn.BatchNorm1d(800, eps=1e-05, momentum=None)

    def forward(self, x):
        x1 = self.internal(self.fc1(x))
        x1 = self.batchnorm1(x1)
        x1 = self.dropout(x1)
        x2 = self.internal(self.fc2(x1))
        x2 = self.batchnorm2(x2)
        x2 = self.dropout(x2)
        x3 = self.internal(self.fc3(x2))
        x3 = self.batchnorm3(x3)
        x3 = self.dropout(x3)
        x4 = self.internal(self.fc4(x3))
        x4 = self.batchnorm4(x4)
        x4 = self.dropout(x4)
        y = self.activation(self.fc5(x4))
        return y
