import os
import glob
from collections import defaultdict
import json
import re

import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import rescale, resize, downscale_local_mean

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torchvision.utils import save_image

import captum.attr as capattr
from utils.plot_img import plot

from utils.img_path_info import img_paths, corres_attr_classes


def main():
    network_name = 'resnet50'
    model = getattr(models, network_name)(pretrained=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    labels_path = '../data/imagenet_class_index.json'
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)

    with open("../data/imagenet_classes.txt", "r") as f:
        labels = [s.strip() for s in f.readlines()]

    transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_height = 224
    flag_save = True

    layer_names = {'conv1': [0],
                   'layer1': [0],
                   'layer2': [0],
                   'layer3': [0, 3],
                   'layer4': [0]}

    # layer_names = {'layer4': [2]}
    for i_img in range(len(img_paths)):
        img_path = img_paths[i_img]
        img_name = os.path.splitext(os.path.basename(img_path))[-2]

        if re.search('for_find_slide_img', img_path):
            img_type = img_path.split('/')[-2]
            img_name = img_type + '-' + img_name

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

        input_image = Image.open(img_path)
        transformed_img = transform(input_image)
        input = transformed_img.unsqueeze(0)
        if torch.cuda.is_available():
            input = input.to('cuda')

        logit_output = model(input)
        output = F.softmax(logit_output, dim=1)
        prediction_logit_np = logit_output[0].cpu().detach().numpy()

        top_prob, top_catid = torch.topk(logit_output[0], 3)
        for i_score in range(top_prob.size(0)):
            print("image name is {}, top-{} class is {}, score is {:.4f}".
                  format(img_name, str(i_score+1), labels[top_catid[i_score]], top_prob[i_score].item()))

        for class_name in corres_attr_classes[i_img]:
            cls_idx = labels.index(class_name)
            print('Target class name is {}, score is {:.4f}'.format(class_name, logit_output[0][cls_idx].item()))

        for layer_name in layer_names.keys():
            layer_indices = layer_names[layer_name]
            for layer_idx in layer_indices:
                if layer_name == 'conv1':
                    layer_module = getattr(model, 'maxpool')
                else:
                    layer_module_temp = getattr(model, layer_name)
                    layer_module = layer_module_temp[layer_idx]
                    if layer_name == 'layer4' and layer_idx == 2:
                        layer_module = layer_module_temp[layer_idx].conv2

                integrated_gradients = capattr.LayerIntegratedGradients(model, layer_module)

                for class_name in corres_attr_classes[i_img]:
                    class_idx = labels.index(class_name)
                    save_path = gen_dir + '/' + class_name + '_' + layer_name + '_' + str(layer_idx) + '.npy'
                    # n_steps_s = [50, 100]
                    # for n_steps in n_steps_s:
                    asNeuronAttrs = integrated_gradients.\
                        attribute(input, target=class_idx, method='gausslegendre', n_steps=30)
                    asNeuronAttrs_np = asNeuronAttrs.squeeze().cpu().detach().numpy()

                    print("step size is {}, logit score is {:.3f}, sum of attr is {:.3f}".
                          format(str(50), prediction_logit_np[class_idx], np.sum(asNeuronAttrs_np)))

                    asNeuronAttrs_hm = np.sum(asNeuronAttrs_np, 0)
                    asNeuronAttrs_hm = asNeuronAttrs_hm * (asNeuronAttrs_hm > np.quantile(asNeuronAttrs_hm, 0.65))

                    resized_asNeuronAttrs_hm = resize(asNeuronAttrs_hm, (image_height, image_height), order=1,
                                                mode='constant', anti_aliasing=False)
                    img_np = np.array(input_image)

                    if flag_save:
                        heatmap_name = class_name + '_' + layer_name + '_' + str(layer_idx) + '.jpg'
                        plot(resized_asNeuronAttrs_hm, save_directory=gen_heatmap_dir, save_img_name=heatmap_name,
                             xi=img_np,
                             cmap='RdBu_r', cmap2='seismic', alpha=0.3, flag_save=flag_save)
                        np.save(save_path, asNeuronAttrs_np)
                    print("Image {}, layer idx {}, class {} is finished".
                          format(img_name, layer_idx, class_name))
                    print()


if __name__ == '__main__':
    main()

