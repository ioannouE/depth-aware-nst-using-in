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
    parser.add_argument("--original-image", type=str, required=True,
                                 help="path to the original content image")
    parser.add_argument("--stylised-image-a",  type=str, required=True,
                                 help="path to the stylised image in 2D")
    parser.add_argument("--stylised-image-b", type=str, required=True,
                                 help="path to the stylised image with depth loss")
    
    args = parser.parse_args()

    device = torch.device("cuda")
    print("Device: ", torch.cuda.get_device_name(0))

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


    img_original = args.original_image
    img_a_st = args.stylised_image_a
    img_b_st = args.stylised_image_b
    
    img_org = utils.load_image(img_original) # cv2.imread(img_original)
    img_org_col = cv2.cvtColor(np.float32(img_org), cv2.COLOR_BGR2RGB)

    img_a = utils.load_image(img_a_st) # cv2.imread(img_a_st)
    img_a_col = cv2.cvtColor(np.float32(img_a), cv2.COLOR_BGR2RGB)

    img_b = utils.load_image(img_b_st) # cv2.imread(img_b_st) 
    img_b_col = cv2.cvtColor(np.float32(img_b), cv2.COLOR_BGR2RGB)

    loss_1, (img_org_depth, img_a_depth) = calculate_loss(img_org_col, img_a_col, midas, transform, device)
    loss_2, (_, img_b_depth) = calculate_loss(img_org_col, img_b_col, midas, transform, device)

    print('Loss 1 - (Original image-Stylised image a) : ', loss_1)
    print('Loss 2 - (Original image-Stylised image b) : ', loss_2)

    # calculate relative depth loss 
    rel_loss_1, (rel_loss_img_org_depth, rel_loss_img_st_depth) = relative_depth_mse_loss(img_org, img_a, device)
    print('Relative depth loss: ', rel_loss_1)

    dpt_loss_1, (dpt_loss_img_org_depth, dpt_loss_img_st_depth) = dpt_depth_mse_loss(img_org, img_a, transform, device)
    print('DPT model depth loss: ', dpt_loss_1)

    # fig_images = [rel_loss_img_org_depth, rel_loss_img_st_depth, img_org_depth, img_a_depth]
    fig_images = [rel_loss_img_org_depth, rel_loss_img_st_depth, dpt_loss_img_org_depth, dpt_loss_img_st_depth]
    minimum_value = 255
    maximum_value = -255
    for img in fig_images:
        if (np.min(img) < minimum_value):
            minimum_value = np.min(img)
        if (np.max(img) > maximum_value):
            maximum_value = np.max(img)
    print(minimum_value, maximum_value)
    print(np.min(rel_loss_img_org_depth), np.max(rel_loss_img_org_depth))
    print(np.min(rel_loss_img_st_depth), np.max(rel_loss_img_st_depth))
    print(np.min(img_org_depth), np.max(img_org_depth))
    print(np.min(img_a_depth), np.max(img_a_depth))

    figure_with_colorbar([rel_loss_img_org_depth, rel_loss_img_st_depth, dpt_loss_img_org_depth, dpt_loss_img_st_depth], minimum_value, maximum_value)
    # figure_with_colorbar([rel_loss_img_org_depth, rel_loss_img_st_depth, img_org_depth, img_a_depth], minimum_value, maximum_value)
    # create_figure([img_org, img_org_depth, img_a, img_a_depth, img_b, img_b_depth])


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def nonlinearity(x):
    threshold = torch.mean(x)
    return torch.where(x>0, 1.5 * x, -1.5 * x)

def figure_with_colorbar(images, min_value, max_value):
    fig, axes = plt.subplots(nrows=2, ncols=2)

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
    plt.savefig('images/depth/visualisation_depth_outputs_no_midas_transform.png')



def create_figure(images):

    for i in range(len(images)):
        images[i] = np.array(images[i])
        images[i] = cv2.resize(images[i].shape[:2], (0, 0), None, .25, .25)
        #print(images[i].shape)
        #images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)

    # row_1 = np.concatenate((images[0][:2], images[1]), axis=1)
    # row_2 = np.concatenate((images[2][:2], images[3]), axis=1)
    # row_3 = np.concatenate((images[4][:2], images[5]), axis=1)
    row_1 = cv2.hconcat([images[0].shape[:2], images[1]])
    row_2 = cv2.hconcat([images[2].shape[:2], images[3]])
    row_3 = cv2.hconcat([images[4].shape[:2], images[5]])

    combined_image = cv2.vconcat([row_1, row_2])
    output_image = cv2.vconcat([combined_image, row_3])

    imshow(output_image); show()  


def calculate_loss(img_org, img_st, midas, transform, device):

    #img_org = utils.load_image(img_original) # cv2.imread(img_ship_original)
    #img_org = cv2.cvtColor(np.float32(img_org), cv2.COLOR_BGR2RGB)
    input_batch_1 = transform(img_org).to(device)

    #img_st = utils.load_image(img_stylised) # cv2.imread(img_ship_original)
    #img_st = cv2.cvtColor(np.float32(img_st), cv2.COLOR_BGR2RGB)
    input_batch_2 = transform(img_st).to(device)

    with torch.no_grad():
        prediction_org = midas(input_batch_1)
        prediction_st = midas(input_batch_2)

        #prediction_org = nonlinearity(prediction_org)
        #prediction_st = nonlinearity(prediction_st)


        prediction_1 = torch.nn.functional.interpolate(
            prediction_org.unsqueeze(1),
            size=img_org.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        prediction_2 = torch.nn.functional.interpolate(
            prediction_st.unsqueeze(1),
            size=img_st.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()


    output_1 = prediction_1.cpu().numpy()
    output_2 = prediction_2.cpu().numpy()
    
    #print(prediction_org.size())
    #print(prediction_org)
    # imshow(output_1); show()
    # imshow(output_2); show()

    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(prediction_org, prediction_st)
    return loss.item(), (output_1, output_2)


########################################################################
def dpt_depth_mse_loss(img_org, img_st, transform, device):

    device = torch.device("cuda")
    print("Device: ", torch.cuda.get_device_name(0))

    # midas = DPTDepthModel()
    # for param in midas.parameters():
    #     param.requires_grad = False
    # midas.to(device)

    # midas.scratch.output_conv[5].register_forward_hook(get_activation('dpt_1'))

    # img_org = utils.load_image(img_original) # cv2.imread(img_ship_original)
    # img_org = cv2.cvtColor(np.float32(img_org), cv2.COLOR_BGR2RGB)
    # input_batch_1 = transform(img_org).to(device)

    # img_st = utils.load_image(img_stylised) # cv2.imread(img_ship_original)
    # img_st = cv2.cvtColor(np.float32(img_st), cv2.COLOR_BGR2RGB)
    # input_batch_2 = transform(img_st).to(device)

    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.to(device)
    midas.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.Resize((orig_width, orig_height)),
		transforms.ToTensor(),
	])
	
    input_batch_1 = transform(img_org).unsqueeze(0).to(device)

    input_batch_2 = transform(img_st).unsqueeze(0).to(device)

    #with torch.no_grad():
    prediction_org = midas(input_batch_1)
    prediction_st = midas(input_batch_2)

    img_org_col = cv2.cvtColor(np.float32(img_org), cv2.COLOR_BGR2RGB)
    img_st_col = cv2.cvtColor(np.float32(img_st), cv2.COLOR_BGR2RGB)
    #prediction_org = activation['dpt_1'].squeeze(dim=1)
    #prediction_st = prediction_st.path_1.squeeze(dim=1)
    
    # print(midas.scratch.output_conv)
    # print(activation['dpt_1'].size())
    print(prediction_org.size())
    #print(prediction_org)
    print(prediction_st.size())

    prediction_1 = torch.nn.functional.interpolate(
        prediction_org.unsqueeze(1),
        size=img_org_col.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    prediction_2 = torch.nn.functional.interpolate(
        prediction_st.unsqueeze(1),
        size=img_st_col.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    output_1 = prediction_1.detach().cpu().numpy()
    output_2 = prediction_2.detach().cpu().numpy()

    # imshow(output_1); show()
    # imshow(output_2); show()

    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(prediction_org, prediction_st)
    return loss.item(), (output_1, output_2)

###########################################################################
def depth_mse_loss(img_org, img_st):

    device = torch.device("cuda")
    print("Device: ", torch.cuda.get_device_name(0))

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

    #img_org = utils.load_image(img_original) # cv2.imread(img_ship_original)
    #img_org = cv2.cvtColor(np.float32(img_org), cv2.COLOR_BGR2RGB)
    input_batch_1 = transform(img_org).to(device)

    #img_st = utils.load_image(img_stylised) # cv2.imread(img_ship_original)
    #img_st = cv2.cvtColor(np.float32(img_st), cv2.COLOR_BGR2RGB)
    input_batch_2 = transform(img_st).to(device)

    with torch.no_grad():
        prediction_org = midas(input_batch_1)
        prediction_st = midas(input_batch_2)

        prediction_1 = torch.nn.functional.interpolate(
            prediction_org.unsqueeze(1),
            size=img_org.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        prediction_2 = torch.nn.functional.interpolate(
            prediction_st.unsqueeze(1),
            size=img_st.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output_1 = prediction_1.cpu().numpy()
    output_2 = prediction_2.cpu().numpy()

    # imshow(output_1); show()
    # imshow(output_2); show()

    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(prediction_org, prediction_st)
    return loss.item() #, (output_1, output_2)


##########################################################################
def relative_depth_mse_loss(img_org, img_st, device):

    orig_width, orig_height = img_org.size

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.Resize((orig_width, orig_height)),
		transforms.ToTensor(),
	])
	

    hourglass = Model()
    #hourglass.seq[3].list[0].register_forward_hook(get_activation('hourglass_1'))
    for param in hourglass.parameters():
        param.requires_grad = False
    # hourglass = hourglass.to(device)  

    
    trained_model_name = 'fast_neural_style/relative_depth/src/results/Best_model2_period1.pt'
    state_dict = torch.load(trained_model_name)

    hourglass.load_state_dict(state_dict)
    hourglass.to(device)
    hourglass.eval()

    #print(hourglass.seq)

    ##input_batch_1 = transform(img_org).unsqueeze(0).to(device) #.float()
    ##input_batch_2 = transform(img_st).unsqueeze(0).to(device) #.float()
    

    # batch_input_org = torch.Tensor(1,3,orig_width,orig_height)
    # batch_input_org = torch.Tensor(1,3,512,512)
    # input_batch_1 = transform(img_org).float()
    # batch_input_org[0,:,:,:] = (input_batch_1)

    # # #processed_input_org = torch.autograd.Variable(batch_input_org.cuda())
    # prediction_org = (hourglass(batch_input_org.cuda())).float()

    # a = prediction_org[0,:,120,:]
	
    # t_back = transforms.Compose([
	# 	transforms.ToPILImage(),
	# 	#transforms.Resize((orig_width, orig_height))
	# ])
    
    # orig_size_output = prediction_org.data[0].cpu()
	# # print(orig_size_output[0,0])
    # orig_size_output = orig_size_output - torch.min(orig_size_output)
    # orig_size_output = orig_size_output / torch.max(orig_size_output)
    # orig_size_output = t_back(orig_size_output)#.convert('RGB')

	# # orig_size_output.save(cmd_params.output_image)
    # new_image = Image.new('RGB', (orig_width*2, orig_height))
    # new_image.paste(img_org, (0,0))
    # new_image.paste(orig_size_output, (orig_width, 0))

    # #imshow(orig_size_output); show()

    # #prediction_org = hourglass(input_batch_1)

    # #input_batch_2 = transform(img_st).float()
    # #prediction_st = hourglass(input_batch_2)


    img_org_col = cv2.cvtColor(np.float32(img_org), cv2.COLOR_BGR2RGB)
    img_st_col = cv2.cvtColor(np.float32(img_st), cv2.COLOR_BGR2RGB)

    input_batch_1 = transform(img_org).unsqueeze(0).to(device) #.float()
    input_batch_2 = transform(img_st).unsqueeze(0).to(device) #.float()

    prediction_org = hourglass(input_batch_1)
    prediction_st = hourglass(input_batch_2)
    
    # prediction_org = activation['hourglass_1'].squeeze(dim=1)
    print(prediction_org.size())
    print(prediction_st.size())

    prediction_1 = torch.nn.functional.interpolate(
        prediction_org[0].unsqueeze(1),
        size=img_org_col.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    prediction_2 = torch.nn.functional.interpolate(
        prediction_st[0].unsqueeze(1),
        size=img_st_col.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    output_1 = prediction_1.cpu().numpy()
    output_2 = prediction_2.cpu().numpy()

    # imshow(output_1); show()
    # imshow(output_2); show()

    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(prediction_org, prediction_st)
    return loss.item(), (output_1, output_2)



if __name__ == "__main__":
    main()