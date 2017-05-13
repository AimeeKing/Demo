import os
import math
import random
import glob
import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.image as mpimg
import sys
sys.path.append('H:\programingProject\Python\TextBox\\')
from ssd_nets import ssd_vgg_300, np_methods
from ssd_preprocessing import ssd_vgg_preprocessing
from ssd_test import visualization

#导入自定义模块
sys.path.append('H:\programingProject\Python\\RCNN\\')

import tensorflow as tf
from crnn_net import model_save as model
from crnn_dataset.utils import  encode_label,sparse_tensor_to_str
import glob
import os
from utils.recognize_utils import load_image_from_dir
checkpoint_dir = os.path.join(sys.path[-1],'crnn_tmp/')


# =========================================================================== #
# recognize
# =========================================================================== #


width_input = tf.placeholder(tf.int32, shape=())
img_input = tf.placeholder(tf.float32, shape=(None, None, 3))
tf.reshape(img_input,shape=(32,-1,3))
img_4d = tf.expand_dims(img_input, 0)


# define the crnn crnn_net
crnn_params = model.CRNNNet.default_params._replace(batch_size=1)  # ,seq_length=int(width/4+1)
crnn = model.CRNNNet(crnn_params)
logits, inputs, seq_len, W, b = crnn.net(img_4d, width=width_input)

decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

saver = tf.train.Saver()

sess = tf.Session()
dir = tf.train.latest_checkpoint(checkpoint_dir)
saver.restore(sess, dir)
sess.run(tf.local_variables_initializer())
print("Model restore!")

def recognize_img(img_dir):
    img_raw, width = load_image_from_dir(img_dir)

    decoded_s = sess.run([decoded],feed_dict={img_input:img_raw,width_input:width})
    # print(decoded_s[0])
    str = sparse_tensor_to_str(decoded_s[0])
    # print("label",label)
    print('识别结果',str)
    return str



# =========================================================================== #
# detect
# =========================================================================== #
sess = tf.Session()

# Input placeholder.
net_shape = (300, 300)
num_classes = 2
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD crnn_net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)# WARP_RESIZE
image_4d = tf.expand_dims(image_pre, 0)
#Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_params = ssd_vgg_300.SSDNet.default_params._replace(num_classes=num_classes)
ssd_net = ssd_vgg_300.SSDNet(ssd_params)

with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False)

checkpoint_dir = os.path.join(sys.path[-1],'synthText/')
saver = tf.train.Saver()
dir = tf.train.latest_checkpoint(checkpoint_dir)

saver.restore(sess, dir)

ssd_anchors = ssd_net.anchors(net_shape)

# Main image processing routine.
def process_image(img, select_threshold=0.45, nms_threshold=.15, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = sess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the crnn_net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)#原来是num_class = 21

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms_fast(rclasses, rscores, rbboxes, nms_threshold=nms_threshold,intersection_threshold = 0.3)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rimg,rclasses, rscores, rbboxes



# Test on some demo image and visualize output.
path = './demo/*.jpg'
image_names = glob.glob(path)
for i,img_name in enumerate(image_names):
      print(i,img_name)
# for i in range(100):
#     i = int(input("choose:\n"))
      img = mpimg.imread(image_names[i])
      rimg,rclasses, rscores, rbboxes =  process_image(img)
      #切片,rimg[:,xmin:xmax,ymin:ymax,:],array
    # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
      visualization.plt_bboxes(img, rclasses, rscores, rbboxes, "./demo/result/%s" % (os.path.basename(image_names[i]).split('.')[0]))
      visualization.crop_by_bboxes(img, rclasses, rscores, rbboxes, "./demo/result/%s" % (os.path.basename(image_names[i]).split('.')[0]))


sess.close()