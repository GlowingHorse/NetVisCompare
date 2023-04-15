# cal PSNR FFT2D LPIPS
import re
import cv2 as cv
import torch
from torch.optim import Adam
from torchvision import models
from torchvision import transforms
import timm_mod
# sample execution (requires torchvision)
from PIL import Image
import numpy as np
import glob
import os
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from scipy.spatial import distance
from utils.img_path_info import img_paths, corres_attr_classes

print()

# Read the categories
with open("../data/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

model_num = 0

if model_num == 0:
    network_name = 'ViT_B_16'
else:
    network_name = 'resnet50'

if network_name.startswith('vit_'):
    model = timm_mod.create_model(network_name, pretrained=True)
    layer_name = 'blocks'
    optimizer_name = 'Adamax'
else:
    model = models.resnet50(pretrained=True)
    layer_name = 'conv1'
    optimizer_name = 'Adamax'

psnr = PeakSignalNoiseRatio()
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

preprocess = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ext = ['png', 'jpg', 'gif', 'jpeg']

for i_img in range(len(img_paths)):
    ori_img_path = img_paths[i_img]
    img_name = os.path.splitext(os.path.basename(ori_img_path))[-2]

    input_image = Image.open(ori_img_path)
    input_image_tensor = preprocess(input_image)

    img = cv.imread(ori_img_path)
    f_ori = np.fft.fft2(img)
    fshift_ori = np.fft.fftshift(f_ori)

    if len(input_image_tensor) == 3:
        # Only one image, so add batch into it
        input_image_tensor = torch.unsqueeze(input_image_tensor, 0)

    if network_name.startswith('vit_'):
        vis_load_dir = '../experiments' + '/' + network_name + \
                       '/img_info/' + img_name + '/' + 'vis_quality_test'

        sub_dirs = os.listdir(vis_load_dir)
        for sub_dir_short in sub_dirs:
            sub_dir = os.path.join(vis_load_dir, sub_dir_short)
            vis_img_paths = []
            [vis_img_paths.extend(glob.glob(sub_dir + '/*.' + e)) for e in ext]
            print('Sub dir name {}'.format(sub_dir_short))

            for vis_img_path in vis_img_paths:
                if re.search(layer_name + '0', vis_img_path):
                    vis_image = Image.open(vis_img_path)
                    vis_image_tensor = preprocess(vis_image)

                    vis_img = np.array(vis_image)
                    # vis_img = cv.imread(vis_img_path)
                    f_vis = np.fft.fft2(vis_img)
                    fshift_vis = np.fft.fftshift(f_vis)

                    fft2d = distance.cosine(fshift_vis.flatten(), fshift_ori.flatten())

                    psnr_res = psnr(vis_image_tensor, input_image_tensor)

                    if len(vis_image_tensor) == 3:
                        vis_image_tensor = torch.unsqueeze(vis_image_tensor, 0)
                    lpips_res = lpips(vis_image_tensor, input_image_tensor)
                    print('PSNR: ', psnr_res)
                    print('FFT2D: ', fft2d)
                    print('LPIPS: ', lpips_res)
                    print()
