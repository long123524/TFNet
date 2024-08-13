import os
import math
import random
import numpy as np
from skimage import io, exposure
from torch.utils import data
from skimage.transform import rescale
from torchvision.transforms import functional as F


num_classes = 1
MEAN = np.array([123.675, 116.28, 103.53])
STD  = np.array([58.395, 57.12, 57.375])
root = r'D:\croplands'

# def showIMG(img):
#     plt.imshow(img)
#     plt.show()
#     return 0

def normalize_image(im):
    #im = (im - MEAN) / STD
    im = im/255
    return im.astype(np.float32)

def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs

def Color2Index(ColorLabel):
    IndexMap = ColorLabel.clip(max=1)
    return IndexMap

def Index2Color(pred):
    pred = exposure.rescale_intensity(pred, out_range=np.uint8)
    return pred

def sliding_crop_CD(imgs1, masks, contours, size):
    crop_imgs1 = []
    crop_imgs2 = []
    crop_labels = []
    label_dims = len(contours[0].shape)
    for img1, mask, contour in zip(imgs1, masks, contours):
        h = img1.shape[0]
        w = img1.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            crop_imgs1.append(img1)
            crop_imgs2.append(mask)
            crop_labels.append(contour)
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times==1: stride_h=0
        else:
            stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))            
        if w_times==1: stride_w=0
        else:
            stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs1.append(img1[s_h:e_h, s_w:e_w, :])
                crop_imgs2.append(mask[s_h:e_h, s_w:e_w])
                if label_dims==2:
                    crop_labels.append(contour[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(contour[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d pairs of images created.' %len(crop_imgs1))
    return crop_imgs1, crop_imgs2, crop_labels

def rand_crop_CD(img1, mask, contour, size):
    # print(img.shape)
    h = img1.shape[0]
    w = img1.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        s_h = random.randint(0, h-c_h)
        e_h = s_h + c_h
        s_w = random.randint(0, w-c_w)
        e_w = s_w + c_w

        crop_im1 = img1[s_h:e_h, s_w:e_w, :]
        mask_im2 = mask[s_h:e_h, s_w:e_w]
        contour_label = contour[s_h:e_h, s_w:e_w]
        # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
        return crop_im1, mask_im2, contour_label

def rand_flip_CD(img1, mask, contour):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img1, mask, contour
    elif r < 0.5:
        return np.flip(img1, axis=0).copy(), np.flip(mask, axis=0).copy(), np.flip(contour, axis=0).copy()
    elif r < 0.75:
        return np.flip(img1, axis=1).copy(), np.flip(mask, axis=1).copy(), np.flip(contour, axis=1).copy()
    else:
        return img1[::-1, ::-1, :].copy(), mask[::-1, ::-1].copy(), contour[::-1, ::-1].copy()

def read_RSimages(mode, read_list=False):
    assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'image')
    mask_dir = os.path.join(root, mode, 'mask')
    contour_dir = os.path.join(root, mode, 'contour')
    
    if mode=='train' and read_list:
        list_path=os.path.join(root, mode+'0.4_info.txt')
        list_info = open(list_path, 'r')
        data_list = list_info.readlines()
        data_list = [item.rstrip() for item in data_list]
    else:
        data_list = os.listdir(img_A_dir)
    data_A, mask_B, contour_labels = [], [], []
    for idx, it in enumerate(data_list):
        if (it[-4:]=='.tif') or (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            mask_path = os.path.join(mask_dir, it)
            contour_path = os.path.join(contour_dir, it)
            
            img_A = io.imread(img_A_path)
            img_A = normalize_image(img_A)
            mask = io.imread(mask_path)
            mask = Color2Index(mask)
            contour = Color2Index(io.imread(contour_path))
            
            data_A.append(img_A)
            mask_B.append(mask)
            contour_labels.append(contour)
        #if idx>10: break    
        if not idx%50: print('%d/%d images loaded.'%(idx, len(data_list)))
    print(data_A[0].shape)
    print(str(len(data_A)) + ' ' + mode + ' images loaded.')   
    return data_A, mask_B, contour_labels

class RS(data.Dataset):
    def __init__(self, mode, random_crop=False, crop_nums=6, sliding_crop=False, crop_size=256, random_flip=True):
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_nums = crop_nums
        self.crop_size = crop_size
        data_A, mask, contour = read_RSimages(mode, read_list=False)
        if sliding_crop:
            data_A, mask, contour = sliding_crop_CD(data_A, mask, contour, [self.crop_size, self.crop_size])
        self.data_A, self.masks, self.contours = data_A, mask, contour
        if self.random_crop:
            self.len = crop_nums*len(self.data_A)
        else:
            self.len = len(self.data_A)

    def __getitem__(self, idx):
        if self.random_crop:
            idx = idx//self.crop_nums
        data_A = self.data_A[idx]
        mask = self.masks[idx]
        boundary = self.contours[idx]
        if self.random_crop:
            data_A, mask, boundary = rand_crop_CD(data_A, mask, boundary, [self.crop_size, self.crop_size])
        if self.random_flip:
            data_A,  mask, boundary = rand_flip_CD(data_A, mask, boundary)
        return F.to_tensor(data_A),  mask, boundary

    def __len__(self):
        return self.len

