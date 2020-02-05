from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
import numpy as np

from colornet import ColorNet
from data import get_training_colorset, get_test_colorset
from tqdm import tqdm

import socket
import time
from tensorboardX import SummaryWriter
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")

########################
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
########################

parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning Rate. Default=0.0002')
parser.add_argument('--lr_step', type=list, default=[200, 400, 500], help='Learning Rate. Default=0.0002')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--lr_dir', type=str, default='')
parser.add_argument('--hr_dir', type=str, default='')

########################
parser.add_argument('--lr_flist', type=str, default='./train_lr.flist')
parser.add_argument('--hr_flist', type=str, default='./train_hr.flist')
########################

parser.add_argument('--file_list', type=str, default='')
parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=0, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='ColorNet')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--prefix', default='_video', help='Location to save checkpoint models')

########################
parser.add_argument('--pretrained_sr', default='color_ColorNet_video_best.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='color_weights/', help='Location to save checkpoint models')
parser.add_argument('--test_flist', type=str, default='./train_lr.flist', help='test_flist')
parser.add_argument('--output', type=str, default='../train_color', help='test_save_folder')
parser.add_argument('--test_only', type=bool, default=False, help='only test')
########################

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)
writer = SummaryWriter('runs_' + opt.model_type)
min_loss = 100
best_epoch = 0
def train(epoch):
    epoch_loss = 0
    epoch_l1 = 0
    epoch_ssim = 0
    epoch_color = 0
    model.train()
    t0 = time.time()
    for iteration, batch in tqdm(enumerate(training_data_loader, 1)):
        input, target = batch[0], batch[1]
        if cuda:
            input = input.cuda()
            target = target.cuda()


        optimizer.zero_grad()
        
        prediction = model(input)
        loss_color = criterion(prediction, target)
        loss = loss_color
        epoch_color += loss_color.item()

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()


    t1 = time.time()


    print("===> Epoch {} Complete: Avg. Loss: {:.4f} L1. Loss: {:.4f} SSIM. Loss: {:.4f} Color. Loss: {:.4f}||| Timer: {:.4f} sec.".format(epoch, epoch_loss / len(training_data_loader), epoch_l1 / len(training_data_loader),
    epoch_ssim / len(training_data_loader), epoch_color / len(training_data_loader),(t1 - t0)))
    avg_loss = (epoch_loss / len(training_data_loader))

    return avg_loss

def convert_shape(img):
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img


def save_img(img, img_name, pred_flag):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

    # save img
    save_dir = os.path.join(opt.output, img_name.split('/')[-2])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = save_dir + '/' + img_name.split('/')[-1]
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])



def eval():
    model.eval()
    for batch in tqdm(testing_data_loader):
        input, file_index = batch[0], batch[1]

        with torch.no_grad():
            input = input.cuda()
            prediction = model(input)

        for i in range(len(file_index)):
            save_img(prediction[i, :, :, :].cpu().data, img_names[file_index[i]], True)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch, best=False):
    os.makedirs(opt.save_folder, exist_ok=True)
    if best:
        model_out_path = opt.save_folder+'color_'+opt.model_type+opt.prefix+"_best.pth"
    else:
        model_out_path = opt.save_folder+'color_'+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_colorset(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.hr_flist, opt.lr_flist, opt.other_dataset, opt.patch_size, opt.future_frame)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
test_set = get_test_colorset(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.test_flist, opt.other_dataset, opt.future_frame)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
img_names = [line.rstrip() for line in open(opt.test_flist)]

print('===> Building model ', opt.model_type)
if opt.model_type == 'ColorNet':
    model = ColorNet()
model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()


print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained or opt.test_only:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()
if not opt.test_only:
    for _ in range(1):
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_step, gamma=0.5)
        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            avg_loss = train(epoch)
            scheduler.step()
            if ((epoch+1) % (opt.snapshots) == 0) and epoch>0:
                checkpoint(epoch)
            if min_loss > avg_loss:
                min_loss = avg_loss
                checkpoint(epoch, best=True)
else:
    eval()