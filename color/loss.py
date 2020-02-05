import torch.nn as nn


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
    def forward(self, x1, x2):
        return self.l1(x1, x2)
