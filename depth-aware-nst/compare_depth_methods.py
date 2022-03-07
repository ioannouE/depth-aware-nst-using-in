import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from transformer_net_depth import TransformerNetDepth
from vgg import Vgg16

from relative_depth.models.hourglass import Model
# from MiDaS.midas.midas_net_custom import MidasNet_small
from MiDaS.midas.dpt_depth import DPTDepthModel
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2

# for depth use MiDaS (https://pytorch.org/hub/intelisl_midas_v2/)
import torchvision.models as models

from decimal import Decimal

import matplotlib; matplotlib.use('agg')
from pylab import *
import matplotlib.pyplot as plt
from matplotlib import pylab 
from PIL import Image 

import urllib.request

def main():
    
    parser = argparse.ArgumentParser(description='parser for calculating depth with MiDaS')
    parser.add_argument("--image", type=str, required=True,
                                 help="path to the image")
   
    args = parser.parse_args()

    device = torch.device("cuda")
    print("Device: ", torch.cuda.get_device_name(0))


    img_original = args.image

    
    img_org = utils.load_image(img_original) # cv2.imread(img_original)
    img_org_col = cv2.cvtColor(np.float32(img_org), cv2.COLOR_BGR2RGB)

    

    img_midas_depth = midas_depth_image(img_org_col, device, args)

    img_relative_depth = relative_depth_image(img_org, device, args)


    fig_images = [img_midas_depth, img_relative_depth]
    minimum_value = 255
    maximum_value = -255
    for img in fig_images:
        if (np.min(img) < minimum_value):
            minimum_value = np.min(img)
        if (np.max(img) > maximum_value):
            maximum_value = np.max(img)


    midas_output_name = 'images/depth_output/midas_no-labels_'+ args.image.replace('images/depth_test/','').replace('.thumb','.png').replace('.jpg','.png')
    single_figure_with_colorbar(img_midas_depth, midas_output_name, minimum_value, maximum_value)

    sidpiw_output_name = 'images/depth_output/sidpiw_no-labels_'+ args.image.replace('images/depth_test/','').replace('.thumb','.png').replace('.jpg','.png')
    single_figure_with_colorbar(img_relative_depth, sidpiw_output_name, minimum_value, maximum_value)

    # figure_with_colorbar([img_midas_depth, img_relative_depth], minimum_value, maximum_value)


def figure_with_colorbar(images, min_value, max_value):
    fig, axes = plt.subplots(nrows=1, ncols=2)

    count = 0
    for ax in axes.flat:
        # imshow(images[count]); show()
        #images[count] = np.array(images[count])
        im = ax.imshow(images[count], vmin=min_value, vmax=max_value) # vmin =0, vmax = 1
        count += 1
        

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()
    plt.savefig('images/depth/comparison_midas_VS_reldepth.png')


def single_figure_with_colorbar(image, fig_name, min_value, max_value):
    # fig = plt.imshow(image, vmin=np.min(image), vmax=np.max(image))
    # plt.colorbar()
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(image) #, vmin=min_value, vmax=max_value) # uncomment for same scale colourbar
    
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # hide axes
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    # uncomment below to add colourbar
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax) # uncomment to add colourbar

    plt.savefig(fig_name, bbox_inches='tight',dpi=300)



def midas_depth_image(img_org, device, args):

    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    input_batch_1 = transform(img_org).to(device)

    with torch.no_grad():
        prediction_org = midas(input_batch_1)

        prediction_1 = torch.nn.functional.interpolate(
            prediction_org.unsqueeze(1),
            size=img_org.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()


    output_1 = prediction_1.cpu().numpy()
    
    # single_figure_with_colorbar(output_1, 'images/depth_output/midas_'+ args.image.replace('images/depth_test/','').replace('.thumb','.png').replace('.jpg','.png'))
    return output_1

##########################################################################
def relative_depth_image(img_org, device, args):

    orig_width, orig_height = img_org.size

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.Resize((orig_width, orig_height)),
		transforms.ToTensor(),
	])
	

    hourglass = Model()
    for param in hourglass.parameters():
        param.requires_grad = False
    
    trained_model_name = 'fast_neural_style/relative_depth/src/results/Best_model2_period1.pt'
    state_dict = torch.load(trained_model_name)

    hourglass.load_state_dict(state_dict)
    hourglass.to(device)
    hourglass.eval()


    img_org_col = cv2.cvtColor(np.float32(img_org), cv2.COLOR_BGR2RGB)

    input_batch_1 = transform(img_org).unsqueeze(0).to(device) #.float()

    prediction_org = hourglass(input_batch_1)
    
    prediction_1 = torch.nn.functional.interpolate(
        prediction_org[0].unsqueeze(1),
        size=img_org_col.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    output_1 = prediction_1.cpu().numpy()

    # imshow(output_1); show()
    # single_figure_with_colorbar(output_1, 'images/depth_output/sidpiw_'+ args.image.replace('images/depth_test/','').replace('.thumb','.png').replace('.jpg','.png'))

    return output_1


if __name__ == "__main__":
    main()