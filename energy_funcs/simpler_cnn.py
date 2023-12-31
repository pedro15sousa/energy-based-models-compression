## PyTorch
import torch
import torch.nn as nn

class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)
    
class SimpleCNNModel(nn.Module):

    def __init__(self, hidden_features=16, out_dim=1, **kwargs):
        super().__init__()
        # Reduced hidden dimensions compared to the original model
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features

        # Simplified series of convolutions and Swish activation functions
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),  # [18x18]
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),  # [9x9]
            Swish(),
            nn.Flatten(),
            nn.Linear(c_hid2*8*8, out_dim)  # Adjusted to match output dimension
        )

    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x