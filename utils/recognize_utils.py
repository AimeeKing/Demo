import sys
#导入自定义模块
sys.path.append('H:\programingProject\Python\\RCNN\\')

import tensorflow as tf
from crnn_net import model_save as model
import Image
from crnn_dataset.utils import  encode_label,sparse_tensor_to_str
import glob
import os
import math

checkpoint_dir = os.path.join(sys.path[-1],'crnn_tmp/')

import numpy as np



def load_image_from_dir(img_dir):
    """
    :param img_dir:
    :return:img_data
     load image and resize it
    """
    img = Image.open(img_dir)
    size = img.size
    width = math.ceil(size[0] * (32 / size[1]))
    img = img.resize([width, 32])
    im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((img.size[1], img.size[0], 3))
    im_arr = im_arr.astype(np.float32) * (1. / 255) - 0.5

    return im_arr,width

def load_image_Img(img):
    """
    :param img_dir:
    :return:img_data
     load image and resize it
    """
    size = img.size
    width = math.ceil(size[0] * (32 / size[1]))
    img = img.resize([width, 32])
    im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((img.size[1], img.size[0], 3))
    im_arr = im_arr.astype(np.float32) * (1. / 255) - 0.5

    return im_arr,width

def load_label_from_img_dir(img_dir):
    # get label from img name
    img_basename = os.path.basename(img_dir)
    (img_name, postfix) = os.path.splitext(img_basename)
    str = img_name.split("_")
    if len(str)>1:
        return str[1]
    else:
        return str[0]

def prepare_data_from_dir(img_dir):
    """
    :param img_dir:
    :return:
    """
    # first load image and label
    image_raw,width = load_image_from_dir(img_dir)
    label = load_label_from_img_dir(img_dir)
    label = label.lower()
    return image_raw,label,width


