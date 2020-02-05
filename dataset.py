import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import os.path
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def difference(hist1,hist2):
    sum1 = 0
    for i in range(len(hist1)):
       if (hist1[i] == hist2[i]):
          sum1 += 1
       else:
           sum1 += 1 - float(abs(hist1[i] - hist2[i]))/ max(hist1[i], hist2[i])
    return sum1/len(hist1)

def load_img(filepath_hr, filepath_lr, nFrames, scale, other_dataset):

    #random.shuffle(seq) #if random sequence
    if other_dataset:
        target = Image.open(filepath_hr).convert('RGB')
        # input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        input = Image.open(filepath_lr).convert('RGB')
    
    return target, input

def load_testimg_future(filepath_lr, nFrames, scale, other_dataset):
    input = Image.open(filepath_lr).convert('RGB')      
    return input

def load_testimg(filepath_lr, nFrames, scale, other_dataset):
    
    #random.shuffle(seq) #if random sequence
    if other_dataset:
        # target = modcrop(Image.open(filepath_hr).convert('RGB'),scale)
        # input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        input = Image.open(filepath_lr).convert('RGB')
    
    return input

def load_img_future(filepath_hr, filepath_lr, nFrames, scale, other_dataset):


    target = Image.open(filepath_hr).convert('RGB')
        # input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
    input = Image.open(filepath_lr).convert('RGB')

    return target, input


def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo)
    iw = iw - (iw%modulo)
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_tar,patch_size, scale, nFrames, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))#[:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))#[:, ty:ty + tp, tx:tx + tp]
    # img_nn = [j.crop((iy,ix,iy + ip, ix + ip)) for j in img_nn] #[:, iy:iy + ip, ix:ix + ip]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar

def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        # img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            # img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            # img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar
    
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def convert2(img):
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,nFrames, upscale_factor, data_augmentation,  hr_list, lr_list, other_dataset, patch_size, future_frame, transform=None):
        super(DatasetFromFolder, self).__init__()
        alist_hr = [line.rstrip() for line in open(hr_list)]
        alist_lr = [line.rstrip() for line in open(lr_list)]
        self.image_filenames_hr = [join(x) for x in alist_hr]
        self.image_filenames_lr = [join(x) for x in alist_lr]
        
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame
        self.repeat = 2

    def __getitem__(self, index):
        index = int(index // self.repeat)
        if self.future_frame:
            target, input = load_img_future(self.image_filenames_hr[index], self.image_filenames_lr[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input = load_img(self.image_filenames_hr[index], self.image_filenames_lr[index], self.nFrames, self.upscale_factor, self.other_dataset)

        if self.patch_size != 0:
            input, target = get_patch(input,target,self.patch_size, self.upscale_factor, self.nFrames)
        
        if self.data_augmentation:
            input, target = augment(input, target)
        # input, target = denoise(input), denoise(target)
        # flow = [get_flow(input,j) for j in neigbor]
            
        # bicubic = rescale_img(input, self.upscale_factor)
        #target_hsv = convert2(target)
        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            # target_hsv = self.transform(target_hsv)
            # bicubic = self.transform(bicubic)
            # flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]

        return input, target#, target_hsv

    def __len__(self):
        return len(self.image_filenames_lr) * self.repeat


class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(file_list)]
        self.image_filenames = [join(x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            input = load_testimg_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            input = load_testimg(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
            
        # flow = [get_flow(input,j) for j in neigbor]
        # input = denoise(input)
        # bicubic = rescale_img(input, self.upscale_factor)
        file_index = index
        if self.transform:
            # target = self.transform(target)
            input = self.transform(input)
            # bicubic = self.transform(bicubic)
            
            # flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]
            
        return input, file_index
      
    def __len__(self):
        return len(self.image_filenames)
