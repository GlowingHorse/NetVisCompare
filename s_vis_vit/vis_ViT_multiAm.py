# ViT inversion

import os
import numpy as np

import timm_mod
import torch
from torchvision import transforms

from utils import transform_robust
from utils_params import random_params

import urllib
from PIL import Image

from utils.vis_ViT_multiAm_class import ViTAMVisualization
from utils.img_path_info import img_paths, corres_attr_classes

from utils.IO import get_proj_abs_dir


if __name__ == '__main__':
    abs_dir = get_proj_abs_dir()
    gen_dir = abs_dir + '/experiments'

    # Param settings
    image_height = 224
    flag_debug = False
    flag_save = False

    flag_save_loss = False

    flag_our_reg = True
    flag_classical_reg = False

    flag_color_space = False

    iteration_steps = 201

    # optimizer_names = ['Adadelta', 'Adagrad', 'Adam', 'AdamW',
    #                    'Adamax', 'ASGD', 'RMSprop', 'Rprop']  # , 'SGD']
    optimizer_names = ['Adamax']
    if flag_our_reg and flag_color_space:
        # opt_lrs = [None]*5
        opt_lrs = [0.003]*5
    elif not flag_our_reg:
        opt_lrs = [0.003] * 5
    elif not flag_color_space:
        opt_lrs = [0.002] * 5

    # loss_names = ['allRepr', '196Patch', 'class1Patch', 'singlePatch', 'singleChannel', 'multiAllRepr']
    loss_names = ['multiAllRepr']

    network_name = 'ViT_B_16'
    AMVisualization = ViTAMVisualization
    model = timm_mod.create_model(network_name, pretrained=True)

    layer_name = 'blocks'
    # layer_name = 'norm'
    # layer_name = 'pre_logits'

    for i_optimizer_name in range(len(optimizer_names)):
        optimizer_name = optimizer_names[i_optimizer_name]
        opt_lr = opt_lrs[i_optimizer_name]
        # for layer_index in range(12):
        for layer_index in [6]:
            # Fully connected layer is not needed
            # pretrained_model = models.googlenet(pretrained=True, transform_input=False)
            for flag_layer in ['expec']:
                patch_index = 1
                # for patch_index in [0, 39, 45, 98, 99, 170, 500, 700, 767]:
                for i_loss in range(len(loss_names)):
                    loss_name = loss_names[i_loss]
                    for i_img in range(len(img_paths)):
                        img_path = img_paths[i_img]
                        input_image = Image.open(img_path)

                        preprocess = transforms.Compose([
                            # transforms.Resize(256),
                            # transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                        input_img = preprocess(input_image)
                        input_img_np = input_img.detach().numpy()
                        corres_class = corres_attr_classes[i_img]

                        img_name = os.path.splitext(os.path.basename(img_path))[-2]

                        layer_vis = AMVisualization(model, input_img_tensor=input_img,
                                                    image_height=image_height,
                                                    layer_name=layer_name, layer_index=layer_index,
                                                    flag_layer=flag_layer,
                                                    gen_dir=gen_dir,
                                                    optimizer_name=optimizer_name, opt_lr=opt_lr,
                                                    loss_name=loss_name, loss_patch_index=patch_index,
                                                    img_name=img_name, network_name=network_name)

                        # Layer visualization with pytorch hooks
                        layer_vis.visualize_fm_with_hooks(flag_debug=flag_debug, flag_save=flag_save,
                                                          flag_our_reg=flag_our_reg, flag_classical_reg=flag_classical_reg,
                                                          flag_color_space=flag_color_space,
                                                          flag_save_loss=flag_save_loss,
                                                          iteration_steps=iteration_steps)
                        print('Img name: {}, opt name: {}, loss_name: {}, layer index: {}, flag_layer: {}'.
                              format(img_name, optimizer_name, loss_name, layer_index, flag_layer))

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
