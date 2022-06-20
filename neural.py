import torch.nn as nn
import torch.nn.functional as F


class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        hidden_layer_1 = 512
        hidden_layer_2 = 256
        self.fc1 = nn.Linear(28*28, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, 10)
        self.dropout=nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)

        return x
