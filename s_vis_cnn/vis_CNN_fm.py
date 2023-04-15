# CNN inversion
import os
import numpy as np

from torchvision import models
from torchvision import transforms

from PIL import Image
from utils.vis_CNN_fm_class import CNNFMVisualization
from utils.img_path_info import img_paths, corres_attr_classes

if __name__ == '__main__':
    # Param settings
    image_height = 224
    flag_debug = False
    flag_save = True
    flag_transparent = False

    iteration_steps = 801

    optimizer_names = ['Adamax']

    network_name = 'resnet50'
    pretrained_model = getattr(models, network_name)(pretrained=True)

    # layer_names = {'conv1': [0],
    #                'layer1': [0, 1, 2],
    #                'layer2': list(range(4)),
    #                'layer3': [0, 1, 2, 3, 4, 5],
    #                'layer4': [0, 1, 2]}
    # opt_lrs = [0.005] * 5
    # layer_names = {'avgpool': [0],
    #                'fc': [0]}
    # layer_names = {'conv1': [0],
    #                'layer1': [0],
    #                'layer2': [0],
    #                'layer3': [0],
    #                'layer4': [0, 1],
    #                'fc': [0]}
    layer_names = {'layer4': [0]}
    opt_lrs = [0.002] * 15

    for i_optimizer_name in range(len(optimizer_names)):
        optimizer_name = optimizer_names[i_optimizer_name]
        opt_lr = opt_lrs[i_optimizer_name]
        for layer_name in layer_names.keys():
            layer_indices = layer_names[layer_name]
            for layer_index in layer_indices:
                for i_img in range(len(img_paths)):
                    img_path = img_paths[i_img]
                    input_image = Image.open(img_path)

                    preprocess = transforms.Compose([
                        # transforms.Resize(256),
                        # transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_img = preprocess(input_image)
                    input_img_np = input_img.detach().numpy()
                    corres_class = corres_attr_classes[i_img]

                    img_name = os.path.splitext(os.path.basename(img_path))[-2]

                    layer_vis = CNNFMVisualization(pretrained_model, input_img_tensor=input_img, image_height=image_height,
                                                   layer_name=layer_name, layer_index=layer_index,
                                                   gen_dir='../experiments',
                                                   optimizer_name=optimizer_name, opt_lr=opt_lr,
                                                   img_name=img_name, network_name=network_name,
                                                   flag_transparent=flag_transparent)

                    # Layer visualization with pytorch hooks
                    layer_vis.visualize_fm_with_hooks(flag_debug=flag_debug, flag_save=flag_save,
                                                      iteration_steps=iteration_steps)
                    print('Img name: {}, opt name: {}, layer_name: {}, layer index: {}'.
                          format(img_name, optimizer_name, layer_name, layer_index))

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
