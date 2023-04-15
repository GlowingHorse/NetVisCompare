

import os
import numpy as np

import torch
from torchvision import models
from torchvision import transforms

from utils import transform_robust
from utils_params import random_params

from utils import misc_functions

import matplotlib
matplotlib.use('TkAgg')
from PIL import Image
from utils.util_loss import InversionLoss
from utils.IO import get_proj_abs_dir

# 对于resnet50, torch名字和经典网络网站名字对应关系
# layer1 = res block2
# layer2 = res block3
# layer3 = res block4
# layer4 = res block5
# block4a = layer3._modules['0'] output


class CNNFMVisualization:
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, input_img_tensor, image_height,
                 layer_name='layer3', layer_index=3,
                 gen_dir=None,
                 optimizer_name='RMSprop', opt_lr=None,
                 img_name=None, network_name='Resnet50',
                 flag_transparent=False):
        self.abs_dir = get_proj_abs_dir()
        self.model = model
        self.model.eval()

        self.input_img_tensor = input_img_tensor

        self.layer_name = layer_name
        self.layer_index = layer_index

        self.image_height = image_height

        self.optimizer_name = optimizer_name
        self.opt_lr = opt_lr

        self.flag_transparent = flag_transparent

        gen_dir = gen_dir + '/' + network_name + '/img_info/' + img_name + '/' + 'visual_results'
        self.gen_dir = gen_dir

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if len(self.input_img_tensor.shape) == 3:
            # Only one image, so add batch into it
            self.input_img_tensor = torch.unsqueeze(self.input_img_tensor, 0)
        if torch.cuda.is_available():
            self.model.to(self.device)
            self.input_img_tensor = self.input_img_tensor.to('cuda')

        self.batch_size = self.input_img_tensor.shape[0]
        # Create the folder to export images if not exists
        if not os.path.exists(self.gen_dir):
            os.makedirs(self.gen_dir)

    def hook_layer(self):
        def hook_function(module, conv_in, conv_out):
            self.conv_outputs = conv_out

        # Hook the selected layer

        layer_module = getattr(self.model, self.layer_name)
        # print('layer name is {}'.format(self.layer_name))
        # for name, module in layer_module.named_modules():
        #     print(name)
        if self.layer_name == 'conv1' or 'avgpool' or 'fc':
            layer_module.register_forward_hook(hook_function)
        else:
            layer_module[self.layer_index].register_forward_hook(hook_function)

        # For Googlenet
        # for name, module in self.model.inception4d.branch4.named_modules():
        #     print(name)
        # self.model.inception4d.register_forward_hook(hook_function)
        # self.model.inception4d.branch3.register_forward_hook(hook_function)
        # self.model[self.selected_layer].register_forward_hook(hook_function)

    def alpha_norm(self, input_matrix, alpha):
        """
            Converts matrix to vector then calculates the alpha norm
        """
        alpha_norm = ((input_matrix.view(-1))**alpha).sum()
        return alpha_norm

    def total_variation_norm(self, input_matrix, beta):
        """
            Total variation norm is the second norm in the paper
            represented as R_V(x)
        """
        to_check = input_matrix[:, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        return total_variation

    def process_to_img(self, shape, sd, decay_power):
        transforms_seq = transforms.Compose([
            random_params.IrfftToImg(shape, sd, decay_power, self.device)
        ])
        return transforms_seq

    def color_space_decorrelate(self, gamma, flag_decorr=None):
        transforms_seq = transforms.Compose([
            transform_robust.ColorDecorrelation(gamma, self.device, flag_decorr),
        ])
        return transforms_seq

    def pre_process_t(self):
        transforms_seq = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return transforms_seq

    def transform_robustness(self):
        # for resnet50
        if self.layer_name == 'layer4' or self.layer_name == 'layer3':
            transforms_seq = transforms.Compose([
                transforms.Pad(16, padding_mode='reflect'),
                transform_robust.RandomCrop(8),
                transform_robust.Closing(device=self.device),
                transform_robust.Opening(device=self.device),
                transform_robust.RandomCrop(2),
                transform_robust.CropOrPadTo(self.image_height)
            ])
        elif self.layer_name == 'avgpool' or self.layer_name == 'fc':
            transforms_seq = transforms.Compose([
                transforms.Pad(12, padding_mode='reflect'),
                transform_robust.RandomCrop(8),
                transform_robust.Closing(device=self.device),
                transform_robust.Opening(device=self.device),
                transform_robust.RandomCrop(2),
                transform_robust.CropOrPadTo(self.image_height),
            ])
        else:
            transforms_seq = transforms.Compose([
                transforms.Pad(6, padding_mode='reflect'),
                transform_robust.RandomCrop(2),
                transform_robust.RandomCrop(1),
                transform_robust.CropOrPadTo(self.image_height)
            ])
        return transforms_seq

    def transform_robustness_latter(self):
        if self.layer_name == 'layer4' or self.layer_name == 'layer3' \
                or self.layer_name == 'avgpool' or self.layer_name == 'fc':
            transforms_seq = transforms.Compose([
                transform_robust.RandomCrop(1),
                transform_robust.CropOrPadTo(self.image_height)
            ])
        else:
            transforms_seq = transforms.Compose([
                transform_robust.CropOrPadTo(self.image_height)
            ])
        return transforms_seq

    def gen_composed_img(self, t_rgb, t_alpha):
        t_bg = random_params.rand_fft_image(t_rgb.shape, sd=0.2, decay_power=1.5, device=self.device)
        t_bg = self.color_space_decorrelate(gamma=False)(t_bg)
        t_composed = t_bg * (1.0 - t_alpha) + t_rgb * t_alpha
        return t_composed, t_alpha

    def visualize_fm_with_hooks(self, flag_debug=False, flag_save=False,
                                iteration_steps=501):
        # Hook the selected layer
        self.hook_layer()

        shape = (self.batch_size, 3, self.image_height, self.image_height)
        sd, decay_power = 0.02, 1.7

        save_dir = self.gen_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        gamma = False
        flag_decorr = 'KLT'
        spectrums = random_params.rand_spectrum(shape, sd=sd, device=self.device)

        # Define optimizer for the image
        if self.opt_lr is None:
            optimizer = getattr(torch.optim, self.optimizer_name)([spectrums])
        else:
            optimizer = getattr(torch.optim, self.optimizer_name)([spectrums], lr=self.opt_lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=35,
                                                               threshold=0.009, min_lr=0.00005)
        with open(self.abs_dir + "/data/imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]

        # Compute feature maps of real images
        ori_score_temp = self.model(self.input_img_tensor)
        ori_score = ori_score_temp.detach().clone()

        top5_prob, top5_catid = torch.topk(ori_score[0], 5)
        # class_name = categories[top5_catid[0]]
        # class_index = top5_catid[0].item()
        for i in range(top5_prob.size(0)):
            print("class is {}, score is {:.4f}".format(categories[top5_catid[i]], top5_prob[i].item()))

        # Compute feature maps of real images
        ori_fms = self.conv_outputs.detach().clone()
        # seventy_quantile = torch.quantile(ori_fms_temp, 0.75)
        # single_zero_tensor = torch.tensor(0., dtype=torch.float32, device=self.device)
        # ori_fms = torch.where(ori_fms_temp > seventy_quantile, ori_fms_temp, single_zero_tensor)

        for_best_loss = torch.tensor(10, dtype=torch.float32, device=self.device)

        save_step = int(iteration_steps/3) - 1

        inversion_loss = InversionLoss('allRepr', 1)
        alpha_tv = torch.tensor(1e-5, dtype=torch.float32, device=self.device)
        alpha_reg_lambda = torch.tensor(1e-6, dtype=torch.float32, device=self.device)

        # 生成保存的图像名字
        generic_im_path = save_dir + \
            '/vis' + \
            '_' + self.layer_name + '_' + str(self.layer_index) + \
            '_' + self.optimizer_name

        # three_fourths_steps = int(3 / 4 * iteration_steps)
        three_fourths_steps = int(iteration_steps)

        for i_iter in range(1, iteration_steps):
            if i_iter % save_step == 0:
                flag_my_pace = False
            else:
                flag_my_pace = False

            if i_iter % 150 == 0:
                flag_my_infopace = True
            else:
                flag_my_infopace = False

            optimizer.zero_grad()

            vis_image_no_rgb = self.process_to_img(shape, sd, decay_power)(spectrums)
            # vis_image = torch.clamp(self.color_space_decorrelate(decorrelate=decorrelate, gamma=gamma,
            #                                                      flag_decorr=flag_decorr)
            #                         (vis_image_no_rgb[:, 0:3, :, :]), 0, 1)
            vis_image = self.color_space_decorrelate(gamma=gamma,
                                                     flag_decorr=flag_decorr)(vis_image_no_rgb[:, 0:3, :, :])
            if i_iter > three_fourths_steps:
                # x_rgb = self.transform_robustness()(vis_image)
                x_rgb = self.transform_robustness_latter()(vis_image)
            else:
                x_rgb = self.transform_robustness()(vis_image)
            x_rgb = self.pre_process_t()(x_rgb)

            output = self.model(x_rgb)
            loss_list = []
            for i_subloss in range(self.batch_size):
                euc_loss = 1e-1 * inversion_loss(self.conv_outputs[i_subloss], self.conv_outputs[i_subloss],
                                                 ori_fms[i_subloss], ori_fms[i_subloss], 0.0)
                # # Calculate alpha regularization
                reg_alpha = alpha_reg_lambda * self.alpha_norm(vis_image[i_subloss], 6)
                # Calculate total variation regularization
                reg_total_variation = alpha_tv * self.total_variation_norm(vis_image[i_subloss], 2)
                loss_list.append(euc_loss + reg_alpha + reg_total_variation)

            loss = sum(loss_list)

            # Backward
            loss.backward()
            optimizer.step()

            scheduler.step(loss)

            if loss < for_best_loss:
                for_best_loss = loss
                vis_image_save = vis_image.clone().detach()

            # Save image
            if (flag_save and flag_my_pace) or (i_iter == (iteration_steps-1)):
                for i_savefig in range(self.batch_size):
                    im_path = generic_im_path + \
                              '_fig' + str(i_savefig) + '.jpg'
                    if self.flag_transparent:
                        vis_rgbd_temp = vis_image_save[i_savefig]
                        vis_rgb_save = 0.55 * (1.0 - vis_rgbd_temp[3:, ...]) + \
                                       vis_rgbd_temp[:3, ...] * vis_rgbd_temp[3:, ...]
                    else:
                        vis_rgb_save = vis_image_save[i_savefig]
                    misc_functions.save_OI_PILImage_float(vis_rgb_save, im_path)

            # Show train information
            if flag_my_infopace:
                loss_value = loss.data.cpu().numpy()
                if len(loss_value.shape) > 0:
                    loss_value = loss_value[0]
                print('Iteration:', str(i_iter),
                      'Loss:', "{0:.4f}".format(loss_value),
                      'LR: {:.9f}'.format(optimizer.param_groups[0]['lr']))
