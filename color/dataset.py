import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import os.path
import cv2
import random
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

def load_color_img(filepath_hr, filepath_lr, nFrames, scale, other_dataset):
    target = Image.open(filepath_hr).convert('RGB')
    a = int(target.size[0] / scale)
    target = target.resize((int(target.size[0]/scale), int(target.size[1]/scale)), Image.BICUBIC)
    input = Image.open(filepath_lr).convert('RGB')

    return target, input

def load_testimg_future(filepath_lr, nFrames, scale, other_dataset):
    tt = int(nFrames/2)
    if other_dataset:
        # target = Image.open(filepath_hr).convert('RGB')
         # input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        input = Image.open(filepath_lr).convert('RGB')
        
        char_len = len(filepath_lr)
        neigbor=[]
        if nFrames%2 == 0:
            seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt,tt+1) if x!=0]
        #random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath_lr.split('/')[-1][:-4])+i
            endstr = len(filepath_lr.split('/')[-1])
            file_name1=filepath_lr[0:char_len-endstr]+str(index1)+'.png'
            
            if os.path.exists(file_name1):
                # temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                temp = Image.open(file_name1).convert('RGB')
                input_hash = input.resize((8,8)).convert('1')
                temp_hash = temp.resize((8,8)).convert('1')
                hist1 = list(input_hash.getdata())
                hist2 = list(temp_hash.getdata())
                if difference(hist1, hist2) > 0.7:
                    neigbor.append(temp)
                else:
                    # print('neigbor frame similarity is not enough')
                    temp=input
                    neigbor.append(temp)
            else:
            
                # print('neigbor frame- is not exist')
                temp=input
                neigbor.append(temp)
            
    else:
        # target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4-tt,5+tt) if x!=4]
        #random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
    return input, neigbor

def load_color_test(filepath_lr, nFrames, scale, other_dataset):
    input = Image.open(filepath_lr).convert('RGB')

    return input


def load_img_future(filepath_hr, filepath_lr, nFrames, scale, other_dataset):
    tt = int(nFrames/2)

    target = Image.open(filepath_hr).convert('RGB')
        # input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
    input = Image.open(filepath_lr).convert('RGB')
    
    char_len = len(filepath_lr)
    neigbor=[]
    if nFrames%2 == 0:
        seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
    else:
        seq = [x for x in range(-tt,tt+1) if x!=0]
    #random.shuffle(seq) #if random sequence
    for i in seq:
        # print(filepath_lr)
        index1 = int(filepath_lr.split('/')[-1][:-4])+i
        endstr = len(filepath_lr.split('/')[-1])
        file_name1=filepath_lr[0:char_len-endstr]+str(index1)+'.png'
        
        if os.path.exists(file_name1):
            # temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
            temp = Image.open(file_name1).convert('RGB')
            input_hash = input.resize((8,8)).convert('1')
            temp_hash = temp.resize((8,8)).convert('1')
            hist1 = list(input_hash.getdata())
            hist2 = list(temp_hash.getdata())
            if difference(hist1, hist2) > 0.7:
                neigbor.append(temp)
            else:
                # print('neigbor frame similarity is not enough')
                temp=input
                neigbor.append(temp)
        else:
            # print('neigbor frame- is not exist')
            temp=input
            neigbor.append(temp)
    return target, input, neigbor


def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo);
    iw = iw - (iw%modulo);
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_tar, img_nn, patch_size, scale, nFrames, ix=-1, iy=-1):
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
    img_nn = [j.crop((iy,ix,iy + ip, ix + ip)) for j in img_nn] #[:, iy:iy + ip, ix:ix + ip]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn, info_patch



def augment_Color(img_in, img_tar, flip_h=True, rot=True):
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

def denoise(img):
    img = np.array(img)
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

class DatasetFromFolder_Color(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, data_augmentation, hr_list, lr_list, other_dataset,
                 patch_size, future_frame, transform=None):
        super(DatasetFromFolder_Color, self).__init__()
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
        self.denoise = False
        self.repeat = 2
    def __getitem__(self, index):

        index = int(index // self.repeat)
        target, input= load_color_img(self.image_filenames_hr[index], self.image_filenames_lr[index],
                                                     self.nFrames, self.upscale_factor, self.other_dataset)


        # if self.patch_size != 0:
        #     input, target, neigbor, _ = get_patch(input, target, neigbor, self.patch_size, self.upscale_factor,
        #                                           self.nFrames)

        if self.data_augmentation:
            input, target, = augment_Color(input, target)


        if self.transform:
            target = self.transform(target)
            input = self.transform(input)


        return input, target

    def __len__(self):
        return len(self.image_filenames_lr) * self.repeat


class DatasetFromFolderTest_Color(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(DatasetFromFolderTest_Color, self).__init__()
        alist = [line.rstrip() for line in open(file_list)]
        self.image_filenames = [join(x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame
        self.denoise = False

    def __getitem__(self, index):

        input = load_color_test(self.image_filenames[index], self.nFrames, self.upscale_factor,self.other_dataset)
        if self.transform:

            input = self.transform(input)


        return input, index

    def __len__(self):
        return len(self.image_filenames)
