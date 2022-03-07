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


# find the style
STYLES = ['composition_vii', 'feathers', 'fire', 'mosaic', 'starry_night', 'the_scream', 'wave']
COMPOSITE_IMAGE = 'composite_image.png'

def main():
    # parse arguements
    parser = argparse.ArgumentParser(description='parser for computing evaluation metrics')
    parser.add_argument("--model-dir", type=str, required=True,
                                 help="directory to saved output images")

    args = parser.parse_args()

    # retrive content images
    content_images = []
    for img in glob.glob('images/content/*'):
        content_images.append(img)
    
    
    # stylised images
    stylised_images = []

    # check if path exists
    if os.path.exists(args.model_dir):
        print('Directory exists: ', args.model_dir)

        for img in glob.glob(args.model_dir + '/*'):
            if (not COMPOSITE_IMAGE in img) and (not 'diff' in img):
                stylised_images.append(img)
    

    # find stylised images with no depth
    stylised_images_no_depth = []
    style = [style for style in STYLES if style in args.model_dir][0]
    out_style_dir = 'images/output/' + style 
    # check if path exists
    if os.path.exists(out_style_dir):
        print('Stylised images with no depth directory: {}\n'.format(out_style_dir))

        for img in glob.glob(out_style_dir + '/*'):
            if (not COMPOSITE_IMAGE in img) and (not 'diff' in img):
                stylised_images_no_depth.append(img)

    # debug
    for c_img, s_img_no_depth, s_img in zip(content_images, stylised_images_no_depth, stylised_images):
        # print(c_img, s_img_no_depth, s_img)

        c_image = cv2.imread(c_img)
        s_image_no_depth = cv2.imread(s_img_no_depth)
        s_image_no_depth = cv2.resize(s_image_no_depth, (c_image.shape[1], c_image.shape[0]), interpolation = cv2.INTER_AREA)
        s_image = cv2.imread(s_img)
        s_image = cv2.resize(s_image, (c_image.shape[1], c_image.shape[0]), interpolation = cv2.INTER_AREA)

        # structural similarity (SSIM)
        ssim_none = ssim(c_image, c_image, data_range=c_image.max() - c_image.min(), multichannel=True)
        ssim_no_depth = ssim(c_image, s_image_no_depth, data_range=c_image.max() - c_image.min(), multichannel=True)
        ssim_depth = ssim(c_image, s_image, data_range=c_image.max() - c_image.min(), multichannel=True)

        # convert to images
        im_c_img = Image.open(c_img)
        im_s_img_no_depth = Image.open(s_img_no_depth)
        im_s_img = Image.open(s_img)
        

        # Histograms
        hist_none = cv2.calcHist([c_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_none = cv2.normalize(hist_none, hist_none).flatten()
        hist_none_score = cv2.compareHist(hist_none, hist_none, cv2.HISTCMP_CORREL) # compare using correlation

        hist_no_depth = cv2.calcHist([s_image_no_depth], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_no_depth = cv2.normalize(hist_no_depth, hist_no_depth).flatten()
        hist_no_depth_score = cv2.compareHist(hist_no_depth, hist_none, cv2.HISTCMP_CORREL) # compare using correlation


        hist_depth = cv2.calcHist([s_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_depth = cv2.normalize(hist_depth, hist_depth).flatten()
        hist_depth_score = cv2.compareHist(hist_depth, hist_none, cv2.HISTCMP_CORREL) # compare using correlation



        # average hashing (aHash)
        aHash_none = imagehash.average_hash(im_c_img)
        aHash_none_dist = 1 - ((aHash_none - aHash_none) / len(aHash_none))

        aHash_no_depth = imagehash.average_hash(im_s_img_no_depth)
        aHash_no_depth_dist = 1 - ((aHash_no_depth - aHash_none) / len(aHash_none))

        aHash_depth = imagehash.average_hash(im_s_img)
        aHash_depth_dist = 1 - ((aHash_depth - aHash_none) / len(aHash_none))


        # difference hashing (dHash)
        dHash_none = imagehash.dhash(im_c_img)
        dHash_none_dist = 1 - ((dHash_none - dHash_none) / len(dHash_none))

        dHash_no_depth = imagehash.dhash(im_s_img_no_depth)
        dHash_no_depth_dist = 1 - ((dHash_no_depth - dHash_none) / len(dHash_none))

        dHash_depth = imagehash.dhash(im_s_img)
        dHash_depth_dist = 1 - ((dHash_depth - dHash_none) / len(dHash_none))

        label = '{:21s}: \t SSIM: {:.4f},\tHist: {:.4f},\taHash: {:.4f},\tdHash: {:.4f}\n'
        print(label.format('Content image', ssim_none, hist_none_score, aHash_none_dist, dHash_none_dist) +
        label.format('Stylised (no depth)', ssim_no_depth, hist_no_depth_score, aHash_no_depth_dist, dHash_no_depth_dist) +
        label.format('Stylised (depth)', ssim_depth, hist_depth_score, aHash_depth_dist, dHash_depth_dist)
        )






if __name__ == "__main__":
    main()