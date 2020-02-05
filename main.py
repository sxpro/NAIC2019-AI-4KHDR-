from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from rcan import RCAN
from data import get_training_set
from loss import Loss
from tqdm import tqdm
import pdb
import socket
import time
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
#################################
parser.add_argument('--batchSize', type=int, default=36, help='training batch size')
#################################
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=1700, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--lr_step', type=list, default=[500, 900, 1300, 1500, 1600, 1700], help='Learning Rate step')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=24, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--lr_dir', type=str, default='')
parser.add_argument('--hr_dir', type=str, default='')
########################
parser.add_argument('--lr_flist', type=str, default='./train_color.flist')
parser.add_argument('--hr_flist', type=str, default='./train_hr.flist')
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
#######################
parser.add_argument('--file_list', type=str, default='')
parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=1)
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='RCAN')
parser.add_argument('--residual', type=bool, default=False)
#######################
parser.add_argument('--pretrained_sr', default='weights/4x_RCAN_image_best.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='_image', help='Location to save checkpoint models')
##model params
parser.add_argument('--n_resgroups', type=int, default=10, help='n_resgroups for rcan')
parser.add_argument('--n_resblocks', type=int, default=23, help='n_resblocks for rcan')
parser.add_argument('--n_feats', type=int, default=64, help='n_feats for rcan')
parser.add_argument('--reduction', type=int, default=16, help='reduction for rcan')
parser.add_argument('--rgb_range', type=int, default=1, help='rgb_range for rcan')
parser.add_argument('--n_colors', type=int, default=3, help='n_colors for rcan')
parser.add_argument('--res_scale', type=int, default=0.1, help='res_scale for rcan')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)
writer = SummaryWriter('runs_' + opt.model_type)
min_loss = 100
best_epoch =0
def train(epoch):
    epoch_loss = 0
    epoch_l1 = 0
    epoch_ssim = 0
    epoch_temp_color = 0
    epoch_temp_l1 = 0
    model.train()

    for iteration, batch in tqdm(enumerate(training_data_loader, 1)):
        input, target = batch[0], batch[1]

        if cuda:
            input = input.cuda()
            target = target.cuda()
       

        optimizer.zero_grad()
        
        prediction = model(input)
        
        loss_l1, loss_ssim = criterion(prediction, target)
        
        epoch_l1 += loss_l1.item()
        epoch_ssim += loss_ssim.item()
       
        total_loss = loss_l1 + loss_ssim
        epoch_loss += total_loss.item()
        total_loss.backward()
        optimizer.step()

    writer.add_scalar('runs/total', epoch_loss / len(training_data_loader), epoch)
    writer.add_scalar('runs/l1', epoch_l1 / len(training_data_loader), epoch)
    writer.add_scalar('runs/ssim', epoch_ssim / len(training_data_loader), epoch)
    print("===> Epoch {} Complete: Avg. Loss: {:.5f} Content Loss: {:.5} SSIM_Loss: {:.5f} temp_color: {:.5f} temp_l1: {:.5f} ".format(epoch, epoch_loss / len(training_data_loader), epoch_l1 / len(training_data_loader),
                                                    epoch_ssim / len(training_data_loader), epoch_temp_color / len(training_data_loader),  epoch_temp_l1 / len(training_data_loader)))
    avg_loss = (epoch_loss / len(training_data_loader))

    return avg_loss

def convert_shape(img):
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch, best=False):
    os.makedirs(opt.save_folder, exist_ok=True)
    if best:
        model_out_path = opt.save_folder + str(
            opt.upscale_factor) + 'x_' + opt.model_type + opt.prefix + "_best.pth"
    else:
        model_out_path = opt.save_folder+str(opt.upscale_factor)+'x_'+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.hr_flist, opt.lr_flist, opt.other_dataset, opt.patch_size, opt.future_frame)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)


print('===> Building model ', opt.model_type)

if opt.model_type == 'RCAN':
    model = RCAN(opt)
model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = Loss()


print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):

        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

print('Pretrained ColorNet is loaded')
if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])



if __name__ == "__main__":
    for _ in range(2):
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_step, gamma=0.5)
        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            avg_loss = train(epoch)
            scheduler.step()

            if (epoch+1) % (opt.snapshots) == 0 and (epoch+1)>0:
                checkpoint(epoch)
            if min_loss > avg_loss:
                min_loss = avg_loss
                checkpoint(epoch, best=True)

    writer.close()
