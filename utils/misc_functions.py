
import os
import copy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
from torch.autograd import Variable
from torchvision import models


def preprocess_image(pil_im, resize_im=True, device='cpu'):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten = im_as_ten.to(device=device)
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)

    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.cpu().data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def recreate_OI_image(im_as_var):
    recreated_im = copy.copy(im_as_var.cpu().data.numpy()[0])
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def save_OI_PILImage_float(im_as_var, im_path):
    # faster than save_OI_image_float
    if isinstance(im_as_var, np.ndarray):
        recreated_im = im_as_var
    else:
        recreated_im = copy.copy(im_as_var.cpu().data.numpy())
    if len(recreated_im.shape) == 4:
        recreated_im = recreated_im[0]

    if len(recreated_im.shape) == 3 and recreated_im.shape[0] == 3:
        recreated_im = np.transpose(recreated_im, (1, 2, 0))
    elif recreated_im.shape[0] == 3:
        recreated_im = np.transpose(recreated_im, (1, 2, 0))

    im = Image.fromarray((recreated_im*255).astype('uint8'))
    im.save(im_path)


def save_OI_image_float(im_as_var, im_path):
    if isinstance(im_as_var, np.ndarray):
        recreated_im = im_as_var
    else:
        recreated_im = copy.copy(im_as_var.cpu().data.numpy())
    if len(recreated_im.shape) == 4:
        recreated_im = recreated_im[0]

    if len(recreated_im.shape) == 3 and recreated_im.shape[0] == 3:
        recreated_im = np.transpose(recreated_im, (1, 2, 0))
    elif recreated_im.shape[0] == 3:
        recreated_im = np.transpose(recreated_im, (1, 2, 0))

    plt.ioff()
    # plt.ion()
    fig = plt.figure(1, figsize=[2.24, 2.24], dpi=100, frameon=False)

    axis = plt.Axes(fig, [0., 0., 1., 1.])
    axis.set_axis_off()
    fig.add_axes(axis)

    axis.imshow(recreated_im)
    plt.savefig(im_path)


def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    example_list = [('../data/images/dog_cat224.jpg', 56)]
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    original_image = Image.open(img_path).convert('RGB')
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
    pretrained_model = models.alexnet(pretrained=True)
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)
