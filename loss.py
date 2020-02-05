from pytorch_msssim import SSIM

import torch.nn as nn

class Loss(nn.modules.loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()
        self.l1 = nn.L1Loss()
        self.criterion_ssim = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3).cuda()

    def forward(self, sr, hr):
        return self.l1(sr, hr)*3, (1 - self.criterion_ssim(sr, hr))*1e-1
