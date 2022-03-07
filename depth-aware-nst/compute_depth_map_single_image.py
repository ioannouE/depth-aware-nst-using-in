import argparse

import numpy as np
import torch

from torchvision import transforms
import torch.onnx

import utils
import cv2


import matplotlib; matplotlib.use('agg')
from pylab import *
import matplotlib.pyplot as plt
from matplotlib import pylab 
from PIL import Image 

import urllib.request

def main():
    
    parser = argparse.ArgumentParser(description='parser for calculating depth with MiDaS')
    parser.add_argument("--input-image", type=str, required=True,
                                 help="path to the image content image")
    parser.add_argument("--output-image",  type=str, required=True,
                                 help="path to output image")
    
    args = parser.parse_args()

    device = torch.device("cuda")
    print("Device: ", torch.cuda.get_device_name(0))


    img_input = args.input_image
    output_path = args.output_image
    
    img_org = utils.load_image(img_input) # cv2.imread(img_original)
    img_org_col = cv2.cvtColor(np.float32(img_org), cv2.COLOR_BGR2RGB)

    img_midas_depth = midas_depth_image(img_org_col, device, args)
    minimum_value = np.min(img_midas_depth)
    maximum_value = np.max(img_midas_depth)

    single_figure_with_colorbar(img_midas_depth, output_path, minimum_value, maximum_value)



def single_figure_with_colorbar(image, fig_name, min_value, max_value):
    # fig = plt.imshow(image, vmin=np.min(image), vmax=np.max(image))
    # plt.colorbar()
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.figure(frameon=False)
    ax = plt.gca()
    im = ax.imshow(image) #, vmin=min_value, vmax=max_value) # uncomment for same scale colourbar
    
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # hide axes
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    # uncomment below to add colorbar
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax) # uncomment to add colourbar

    plt.axis('off')
    plt.savefig(fig_name, bbox_inches='tight',transparent=True, pad_inches=0, dpi=300)

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
    
    return output_1



if __name__ == "__main__":
    main()