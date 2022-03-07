import argparse
import os
import sys

import numpy as np
import torch

import utils
import cv2

import matplotlib.pyplot as plt
from matplotlib import pylab
from pylab import *
from PIL import Image

import fast_neural_style
import calculate_depth_loss

import glob

from skimage.metrics import structural_similarity as ssim
from SSIM_PIL import compare_ssim
import imagehash


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)
    # return cv2.vconcat(im_list)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


# find the style
STYLES = ['composition_vii', 'feathers', 'fire', 'mosaic', 'starry_night', 'the_scream', 'wave']
COMPOSITE_IMAGE = 'composite_image.png'

def main():
    # parse arguements
    parser = argparse.ArgumentParser(description='parser for computing evaluation metrics')
    parser.add_argument("--image-a", type=str, required=True,
                                 help="path to image a")
    parser.add_argument("--image-b", type=str, required=True,
                                 help="path to image b")
    parser.add_argument("--image-c", type=str, required=True,
                                 help="path to image c")
    parser.add_argument("--image-d", type=str, required=True,
                                 help="path to image d")
    parser.add_argument("--image-e", type=str, required=True,
                                 help="path to image e")
    parser.add_argument("--image-f", type=str, required=True,
                                 help="path to image f")
   
                                 

    args = parser.parse_args()

    image_rows = []

    img_a = args.image_a
    img_b = args.image_b
    img_c = args.image_c
    img_d = args.image_d
    img_e = args.image_e
    img_f = args.image_f


    image_a = cv2.imread(img_a)
    
    image_b = cv2.imread(img_b)
    image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]), interpolation = cv2.INTER_AREA)
    
    image_c = cv2.imread(img_c)
    image_c = cv2.resize(image_c, (image_a.shape[1], image_a.shape[0]), interpolation = cv2.INTER_AREA)

    image_d = cv2.imread(img_d)
    image_d = cv2.resize(image_d, (image_a.shape[1], image_a.shape[0]), interpolation = cv2.INTER_AREA)

    image_e = cv2.imread(img_e)
    image_e = cv2.resize(image_e, (image_a.shape[1], image_a.shape[0]), interpolation = cv2.INTER_AREA)

    image_f = cv2.imread(img_f)
    image_f = cv2.resize(image_f, (image_a.shape[1], image_a.shape[0]), interpolation = cv2.INTER_AREA)

    
    

    # for i, col in enumerate(['b', 'g', 'r']):
    #     hist = cv2.calcHist([image_a], [i], None, [256], [0, 256])
    #     plt.plot(hist, color = col)
    #     plt.xlim([0, 256])
    # plt.savefig("images/other/mygraph.png")
 
    # Histograms
    # hist_a = cv2.calcHist([image_a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # #hist_a = cv2.normalize(hist_a, hist_a).flatten()
    # hist_a_score = cv2.compareHist(hist_a, hist_a, cv2.HISTCMP_CORREL) # compare using correlation
    for i, col in enumerate(['r', 'g', 'b']):
        hist_a = cv2.calcHist([image_a], [i], None, [256], [0, 256])
        plt.plot(hist_a, color = col)
        plt.xlim([0, 256])
        plt.ylim([0, 5000])
    plt.savefig("images/histograms/1/hist_a.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
    hist_a_img = cv2.imread('images/histograms/1/hist_a.png')



    for i, col in enumerate(['r', 'g', 'b']):
        hist_b = cv2.calcHist([image_b], [i], None, [256], [0, 256])
        plt.plot(hist_b, color = col)
        plt.xlim([0, 256])
        plt.ylim([0, 5000])
    plt.savefig("images/histograms/1/hist_b.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
    hist_b_img = cv2.imread('images/histograms/1/hist_b.png')



    ###### results histograms ######
    for i, col in enumerate(['r', 'g', 'b']):
        hist_c = cv2.calcHist([image_c], [i], None, [256], [0, 256])
        plt.plot(hist_c, color = col)
        plt.xlim([0, 256])
        plt.ylim([0, 5000])
    plt.savefig("images/histograms/1/hist_c.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
    hist_c_img = cv2.imread('images/histograms/1/hist_c.png')


    for i, col in enumerate(['r', 'g', 'b']):
        hist_d = cv2.calcHist([image_d], [i], None, [256], [0, 256])
        plt.plot(hist_d, color = col)
        plt.xlim([0, 256])
        plt.ylim([0, 5000])
    plt.savefig("images/histograms/1/hist_d.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
    hist_d_img = cv2.imread('images/histograms/1/hist_d.png')


    for i, col in enumerate(['r', 'g', 'b']):
        hist_e = cv2.calcHist([image_e], [i], None, [256], [0, 256])
        plt.plot(hist_e, color = col)
        plt.xlim([0, 256])
        plt.ylim([0, 5000])
    plt.savefig("images/histograms/1/hist_e.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
    hist_e_img = cv2.imread('images/histograms/1/hist_e.png')

    for i, col in enumerate(['r', 'g', 'b']):
        hist_f = cv2.calcHist([image_f], [i], None, [256], [0, 256])
        plt.plot(hist_f, color = col)
        plt.xlim([0, 256])
        plt.ylim([0, 5000])
    plt.savefig("images/histograms/1/hist_f.png",bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
    hist_f_img = cv2.imread('images/histograms/1/hist_f.png')


    image_a = cv2.resize(image_a, (image_a.shape[1], image_a.shape[0]-180), interpolation = cv2.INTER_AREA)
    image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]-180), interpolation = cv2.INTER_AREA)
    image_c = cv2.resize(image_c, (image_a.shape[1], image_a.shape[0]-180), interpolation = cv2.INTER_AREA)
    image_d = cv2.resize(image_d, (image_a.shape[1], image_a.shape[0]-180), interpolation = cv2.INTER_AREA)
    image_e = cv2.resize(image_e, (image_a.shape[1], image_a.shape[0]-180), interpolation = cv2.INTER_AREA)
    image_f = cv2.resize(image_f, (image_a.shape[1], image_a.shape[0]-180), interpolation = cv2.INTER_AREA)


    image_rows.append(hconcat_resize_min([image_b, image_c, image_d, image_e, image_f]))
    image_rows.append(hconcat_resize_min([hist_b_img, hist_c_img, hist_d_img, hist_e_img, hist_f_img]))

    out_img = vconcat_resize_min(image_rows)
    cv2.imwrite('images/histograms/1/all.png', out_img)






if __name__ == "__main__":
    main()