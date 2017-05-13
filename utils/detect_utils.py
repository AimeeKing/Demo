import matplotlib.image as mpimg
import os

import sys
sys.path.append('H:\programingProject\Python\TextBox\\')

from ssd_test import visualization

def save_detected_pic(img_dir,rclasses, rscores, rbboxes):
    img = mpimg.imread(img_dir)
    visualization.plt_bboxes(img, rclasses, rscores, rbboxes,
                             "./static/result/complete/%s" % (os.path.basename(img_dir).split('.')[0]))
    visualization.crop_by_bboxes(img, rclasses, rscores, rbboxes,
                                 "./static/result/clip/%s" % (os.path.basename(img_dir).split('.')[0]))
    return "../static/result/complete/%s" % (os.path.basename(img_dir).split('.')[0])
