''' 
based on utils of PyTorch implementation of Johnson et al 
https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
'''
import torch
from PIL import Image


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int (img.size[1] / scale)), Image.ANTIALIAS)
    
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)

    return gram


def normalize_batch(batch):
    # normalises using imagenet and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.255]).view(-1, 1, 1)
    batch = batch.div_(255.0)

    return (batch - mean) / std


import os
from torch.utils.data import Dataset

folder_names = ['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern', 'Baroque', 
'Color_Field_Painting', 'Contemporary_Realism', 'Cubism', 'Early_Renaissance', 'Expressionism', 'Fauvism', 
'High_Renaissance', 'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism', 'Naive_Art_Primitivism', 
'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art', 'Post_Impressionism', 'Realism', 'Rococo', 
'Romanticism', 'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e']

class ImageDataSet(Dataset):

    def __init__(self, root='../wikiart', image_loader=None, transform=None):
        self.root = root
        self.image_files = [os.listdir(os.path.join(self.root, dir_name)) for dir_name in folder_names]
        self.loader = image_loader
        self.transform = transform
    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        return sum([len(folder) for folder in self.image_files])

    def __getitem__(self, index):
        images = [self.loader(os.path.join(self.root, dir_name, self.image_files[dir_name][index])) for dir_name in folder_names]
        if self.transform is not None:
            images = [self.transform(img) for img in images]
        return images