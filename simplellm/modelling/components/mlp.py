from torch import nn

class MLP(nn.Module):
    def __init__(self, in_channels, fan_out=4, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, fan_out * in_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(fan_out * in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x