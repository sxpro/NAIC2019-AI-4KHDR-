from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from rcan import RCAN

from data import get_test_set
from functools import reduce
import numpy as np
from tqdm import tqdm
import time
import cv2
import math
import pdb

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
#################################
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
#################################
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--file_list', type=str, default='foliage.txt')
parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=1)
parser.add_argument('--model_type', type=str, default='RCAN')
parser.add_argument('--residual', type=bool, default=False)
#####################################
parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
parser.add_argument('--model', default='weights/4x_RCAN_image_best.pth', help='sr pretrained base model')
parser.add_argument('--test_flist', type=str, default='./test_color2.flist')
parser.add_argument('--self_ensemble', type=bool, default=True)
#####################################
##model params for rcan
parser.add_argument('--n_resgroups', type=int, default=10, help='n_resgroups for rcan')
parser.add_argument('--n_resblocks', type=int, default=23, help='n_resblocks for rcan')
parser.add_argument('--n_feats', type=int, default=64, help='n_feats for rcan')
parser.add_argument('--reduction', type=int, default=16, help='reduction for rcan')
parser.add_argument('--rgb_range', type=int, default=1, help='rgb_range for rcan')
parser.add_argument('--n_colors', type=int, default=3, help='n_colors for rcan')
parser.add_argument('--res_scale', type=int, default=0.1, help='res_scale for rcan')
opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_test_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.test_flist, opt.other_dataset, opt.future_frame)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
img_names = [line.rstrip() for line in open(opt.test_flist)]
print('===> Building model ', opt.model_type)

if opt.model_type == 'RCAN':
    model = RCAN(opt)
if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')
if cuda:
    model = model.cuda(gpus_list[0])

def eval():
    model.eval()


    for batch in tqdm(testing_data_loader):
        input, file_index = batch[0], batch[1]
        
        with torch.no_grad():
            input = Variable(input).cuda()

            # flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, model, opt.upscale_factor)
        else:
            with torch.no_grad():
                input = colornet(input)
                prediction = model(input)

            

        # print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
        for i in range(len(file_index)):
            save_img(prediction[i, :, :, :].cpu().data, img_names[file_index[i]], True)


def _transform(v, op):
    v2np = v.data.cpu().numpy()
    if op == 'v':
        tfnp = v2np[:, :, :, ::-1].copy()
    elif op == 'h':
        tfnp = v2np[:, :, ::-1, :].copy()
    elif op == 't':
        tfnp = v2np.transpose((0, 1, 3, 2)).copy()

    ret = torch.Tensor(tfnp).to(torch.device('cuda'))
    return ret   
def eval_ensemble():
    model.eval()
    count=1

    # avg_psnr_predicted = 0.0
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            input, file_index = batch[0], batch[1]
            input = Variable(input).cuda()
            inputa = _transform(input, 'v')
            #inputb = _transform(input, 'h')
            #inputc = _transform(input, 't')

            # t0 = time.time()
            if opt.chop_forward:
                
                prediction = chop_forward(input, model, opt.upscale_factor)
                predictiona = chop_forward(inputa,  model, opt.upscale_factor)
                #predictionb = chop_forward(inputb,  model, opt.upscale_factor)
                #predictionc= chop_forward(inputc, model, opt.upscale_factor)
            else:
                
                prediction = model(input) 
                predictiona = model(inputa) 
                #predictionb = model(inputb) 
                #predictionc = model(inputc) 
            prediction = prediction + _transform(predictiona, 'v') #+ _transform(predictionb, 'h')+ _transform(predictionc, 't')
            prediction /= 2.0
            
        for i in range(len(file_index)):
            save_img(prediction[i, :, :, :].cpu().data, img_names[file_index[i]], True)
        # save_img(prediction.cpu().data, img_names[file_index], True)
        # count+=1


def save_img(img, img_name, pred_flag):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    # save img
    save_dir=os.path.join(opt.output, img_name.split('/')[-2])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_fn = save_dir +'/'+ img_name.split('/')[-1]
    cv2.imwrite(save_fn,  cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

    
def chop_forward(x, model, scale, shave=8, min_size=2000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        [x[:, :, 0:h_size, 0:w_size]],
        [x[:, :, 0:h_size, (w - w_size):w]],
        [x[:, :, (h - h_size):h, 0:w_size]],
        [x[:, :, (h - h_size):h, (w - w_size):w]]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = inputlist[i]#torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                output_batch = model(input_batch[0])
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch[0], model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

##Eval Start!!!!
if __name__ == "__main__":
    
    if opt.self_ensemble:
        eval_ensemble()
    else:
        eval()
