import os
import glob
import re
from collections import defaultdict
import json

import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import rescale, resize, downscale_local_mean

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torchvision.utils import save_image

import timm_mod

import captum.attr as capattr
from utils.plot_img import plot

from utils.img_path_info import img_paths, corres_attr_classes

transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with open("../data/imagenet_classes.txt", "r") as f:
    labels = [s.strip() for s in f.readlines()]

labels_path = '../data/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)


def get_save_dir(img_name, network_name):
    gen_dir = '../experiments/' + network_name + \
              '/img_info/' + img_name + \
              '/' + 'neuronAttr'
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    gen_heatmap_dir = '../experiments/' + network_name + \
                      '/img_info/' + img_name + \
                      '/' + 'neuronAttrHMs'
    if not os.path.exists(gen_heatmap_dir):
        os.makedirs(gen_heatmap_dir)
    return gen_dir, gen_heatmap_dir


def get_model_input(img_path):
    input_image = Image.open(img_path)
    transformed_img = transform(input_image)
    input_tensor = transformed_img.unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
    return input_tensor, input_image


def get_save_path_neuron_attrs(model, class_name, flag_norm2, gen_dir, layer_name,
                               layer_idx, attr_method_name, input_tensor, layer_masks):
    if layer_idx == 11:
        layer_module_temp = getattr(model, 'norm')
        layer_module = layer_module_temp
    else:
        layer_module_temp = getattr(model, layer_name)
        if not flag_norm2:
            layer_module = layer_module_temp[layer_idx]
        else:
            layer_module = layer_module_temp[layer_idx].norm2

    class_idx = labels.index(class_name)

    integrated_gradients = capattr.LayerIntegratedGradients(model, layer_module)
    layerLRP = capattr.LayerLRP(model, layer_module)
    ablation_feature = capattr.LayerFeatureAblation(model, layer_module)

    if attr_method_name == 'IG':
        neuronAttrs = integrated_gradients. \
            attribute(input_tensor, target=class_idx, method='gausslegendre', n_steps=20)
        if not flag_norm2:
            save_path = gen_dir + '/' + class_name + '_' + layer_name + str(layer_idx) + '.npy'
            heatmap_name = class_name + '_' + layer_name + str(layer_idx) + '.jpg'
        else:
            save_path = gen_dir + '/' + class_name + '_' + layer_name + str(layer_idx) + 'norm2.npy'
            heatmap_name = class_name + '_' + layer_name + str(layer_idx) + 'norm2.jpg'
    elif attr_method_name == 'LRP':
        neuronAttrs = layerLRP.attribute(input_tensor, target=class_idx)
        if not flag_norm2:
            save_path = gen_dir + '/' + class_name + '_' + layer_name + str(layer_idx) + '_lrp.npy'
            heatmap_name = class_name + '_' + layer_name + str(layer_idx) + '_lrp.jpg'
        else:
            save_path = gen_dir + '/' + class_name + '_' + layer_name + str(layer_idx) + 'norm2_lrp.npy'
            heatmap_name = class_name + '_' + layer_name + str(layer_idx) + 'norm2_lrp.jpg'
    elif attr_method_name == 'Ablation':
        neuronAttrs = ablation_feature.attribute(input_tensor, target=class_idx, layer_mask=layer_masks)
        neuronAttrs = neuronAttrs[0]
        if not flag_norm2:
            save_path = gen_dir + '/' + class_name + '_' + layer_name + str(layer_idx) + '_abla.npy'
        else:
            save_path = gen_dir + '/' + class_name + '_' + layer_name + str(layer_idx) + 'norm2_abla.npy'
        heatmap_name = None
    neuron_attrs_np = neuronAttrs.squeeze().cpu().detach().numpy()

    return save_path, heatmap_name, neuron_attrs_np


def main():
    network_name = 'ViT_B_16'

    model = timm_mod.create_model(network_name, pretrained=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    image_height = 224
    flag_save = True

    attr_method_name = 'IG'  # 'IG'  'Ablation'  'LRP'

    # modify \TorchVis\Lib\site-packages\captum\attr\_core\layer\layer_integrated_gradients.py
    # grads = torch.autograd.grad(torch.unbind(output), inputs)
    # if type(inputs) == tuple:
    #     if inputs[1].shape[-1] == 197:
    #         grads = torch.autograd.grad(torch.unbind(output), inputs[0])
    # else:
    #     grads = torch.autograd.grad(torch.unbind(output), inputs)

    layer_name = 'blocks'
    used_layer_indices = list(range(11))
    # used_layer_indices = [11]
    flag_norm2s = [False]
    # flag_norm2s = [False]

    layer_mask1 = torch.tensor([[list(range(768))]], device=device)
    layer_mask2 = torch.tensor([[list(range(197))]], device=device)
    layer_masks = (layer_mask1, layer_mask2)

    for i_img in range(len(img_paths)):
        img_path = img_paths[i_img]
        img_name = os.path.splitext(os.path.basename(img_path))[-2]

        if re.search('for_find_slide_img', img_path):
            img_type = img_path.split('/')[-2]
            img_name = img_type + '-' + img_name

        gen_dir, gen_heatmap_dir = get_save_dir(img_name, network_name)
        input_tensor, input_image = get_model_input(img_path)

        logit_output = model(input_tensor)
        output = F.softmax(logit_output, dim=1)
        prediction_logit_np = logit_output[0].cpu().detach().numpy()

        top_prob, top_catid = torch.topk(logit_output[0], 5)
        print()
        for i_score in range(top_prob.size(0)):
            print("image name is {}, top-{} class is {}, score is {:.4f}".
                  format(img_name, str(i_score+1), labels[top_catid[i_score]], top_prob[i_score].item()))

        for layer_idx in used_layer_indices:
            for flag_norm2 in flag_norm2s:
                for class_name in corres_attr_classes[i_img]:
                    class_idx = labels.index(class_name)

                    save_path, heatmap_name, neuron_attrs_np = \
                        get_save_path_neuron_attrs(model, class_name, flag_norm2, gen_dir, layer_name,
                                                   layer_idx, attr_method_name, input_tensor,
                                                   layer_masks)

                    print("step size is {}, logit score is {:.3f}, sum of attr is {:.3f}".
                          format(str(50), prediction_logit_np[class_idx], np.sum(neuron_attrs_np[1:, ])))

                    if flag_save:
                        if attr_method_name != 'Ablation':
                            neuron_attrs_hm_info = neuron_attrs_np[1:, ]
                            neuron_attrs_hm = np.sum(neuron_attrs_hm_info, 1)
                            neuron_attrs_hm = neuron_attrs_hm * (neuron_attrs_hm > np.quantile(neuron_attrs_hm, 0.65))

                            reshaped_neuron_attrs_hm = np.reshape(neuron_attrs_hm, (14, 14))
                            resized_neuron_attrs_hm = resize(reshaped_neuron_attrs_hm, (image_height, image_height),
                                                             order=1,
                                                             mode='constant', anti_aliasing=False)

                            img_np = np.array(input_image)
                            plot(resized_neuron_attrs_hm, save_directory=gen_heatmap_dir,
                                 save_img_name=heatmap_name, xi=img_np,
                                 cmap='RdBu_r', cmap2='seismic', alpha=0.3, flag_save=flag_save)
                        np.save(save_path, neuron_attrs_np)
                    print("Image {}, layer idx {}, class {} is finished".
                          format(img_name, layer_idx, class_name))
                    print()


if __name__ == '__main__':
    main()

