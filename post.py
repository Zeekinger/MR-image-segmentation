#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 20:25
import numpy as np
import os
from matplotlib import pyplot as plt
import nibabel as nib
from PIL import Image
import scipy as sp
from skimage import io
from skimage.transform import resize

def print_images(case_id, data_patch, seg_patch, mask_patch,img_path):
    plt.figure(figsize=(35, 15))
    volume_shape = data_patch.shape
    for i in range(volume_shape[2]):

        plt.imshow(mask_patch[..., i], 'gray')
        plt.savefig(os.path.join(os.path.join(img_path, case_id), str(i)+str('专家')  ))
        plt.imshow(data_patch[..., i], 'gray')
        plt.savefig(os.path.join(os.path.join(img_path, case_id), str(i)+str('原始')))
        #plt.subplot(1,2,2)
        plt.imshow(seg_patch[..., i], 'gray')
        plt.savefig(os.path.join(os.path.join(img_path, case_id), str(i)+str('预测')))
    plt.close('all')

def blend_two_images(img_path1, img_path2):
    img1 = Image.open(img_path1)

    img2 = Image.open(img_path2)

    img = Image.blend(img1, img2, 0.5)
    img3=img.crop((1200, 400, 2400, 1100))
    return img3

def postprocess_images(img_path,save_path):     ##（编号，数据补丁，分割补丁，图像路径）
    data_patch = nib.load(img_path).get_data()
    print('A',data_patch.shape)
    affine = nib.load(img_path).get_affine()
    volume_shape = data_patch.shape
    temp = np.zeros((880,880,volume_shape[2]), dtype=np.float32)

    for i in range(volume_shape[2]):
        temp[...,i] = resize(data_patch[..., i], (880, 880), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
    nii_volume = nib.Nifti1Image(temp, affine)
    print(nii_volume.shape)
    nib.save(nii_volume, os.path.join(save_path, 'postprocess.nii'))

