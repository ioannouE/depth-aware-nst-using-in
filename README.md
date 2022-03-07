# Depth-aware Neural Style Transfer using Instance Normalization
Neural Style Transfer (NST) is concerned with the artistic stylization of visual media. It can be described as the process of transferring the style of an artistic image onto an ordinary photograph. Recently, a number of studies have considered the enhancement of the depth-preserving capabilities of the NST algorithms to address the undesired effects that occur when the input content images include numerous objects at various depths. Our approach uses a deep residual convolutional network with instance normalization layers that utilizes an advanced depth prediction network to integrate depth preservation as an additional loss function to content and style. We demonstrate results that are effective in retaining the depth and global structure of content images.

## Setup
* [PyTorch](http://pytorch.org/) (version used: torch 1.8.1, torchvision 0.9.1)
* [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) (version used: 11.0)
* For tracking experiments: [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) (version used: 2.7.0)


 ## Usage
Stylize image
```
python depth-aware-nst/fast_neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --cuda 0
```
* `--content-image`: path to content image you want to stylize.
* `--model`: saved model to be used for stylizing the image (eg: `mosaic.pth`)
* `--output-image`: path for saving the output image.
* `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Train model
```bash
python depth-aware-nst/fast_neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --save-model-dir </path/to/save-model/folder> --epochs 2 --cuda 1
```

There are several command line arguments, the important ones are listed below
* `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. I used COCO 2014 Training images dataset [80K/13GB] [(download)](https://cocodataset.org/#download).
* `--style-image`: path to style-image.
* `--save-model-dir`: path to folder where trained model will be saved.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.
* `--content-weight`: weight for content-loss, default is 1e5.
* `--style-weight`: weight for style-loss, default is 1e10.
* `--depth-loss`: set it to 1 to train with depth loss, 0 train without depth loss, default is 1.
* `--depth-weight`: weight for depth-loss, default is 1e10

### Code implementation is influenced by:
* [Structure-Preserving Neural Style Transfer](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8816670) -- [code](https://github.com/xch-liu/structure-nst)
* [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) -- [code](https://github.com/pytorch/examples/tree/master/fast_neural_style)

