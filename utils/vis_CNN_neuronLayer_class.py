
import os
import numpy as np

import torch
from torchvision import transforms

from utils import transform_robust
from utils_params import random_params

from utils import misc_functions
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import math


class CNNLayerVisualization:
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_filters=None, image_height=224,
                 layer_name='layer3', layer_index=3,
                 gen_dir=None,
                 optimizer_name='RMSprop', opt_lr=None,
                 network_name='resnet50',
                 method_vis_neruon=False):

        self.model = model
        self.model.eval()

        self.selected_filters = selected_filters

        self.layer_name = layer_name
        self.layer_index = layer_index

        self.batch_size = len(self.selected_filters)

        self.image_height = image_height

        self.optimizer_name = optimizer_name
        self.opt_lr = opt_lr

        self.method_vis_neruon = method_vis_neruon

        self.gen_dir = gen_dir + '/' + network_name + '/' + \
                  'channel-level-vis' + '/' + \
                  layer_name + '_' + str(layer_index)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            self.model.to(self.device)

        if not os.path.exists(self.gen_dir):
            os.makedirs(self.gen_dir)

    def hook_layer(self, hook_batch, hook_method_vis_neruon, hook_selected_filters):
        def hook_function(module, grad_in, grad_out):
            self.conv_outputs = []
            for i_batch_size in range(hook_batch):
                self.conv_outputs.append(grad_out[i_batch_size, hook_selected_filters[i_batch_size]])

        if self.layer_name == 'conv1':
            layer_module = getattr(self.model, 'maxpool')
            layer_module.register_forward_hook(hook_function)
        else:
            layer_module = getattr(self.model, self.layer_name)
            layer_module[self.layer_index].register_forward_hook(hook_function)

        # For Googlenet
        # for name, module in self.model.inception4d.branch4.named_modules():
        #     print(name)
        # self.model.inception4d.register_forward_hook(hook_function)
        # self.model.inception4d.branch3.register_forward_hook(hook_function)
        # self.model[self.selected_layer].register_forward_hook(hook_function)

    def process_to_img(self, shape, sd, decay_power):
        transforms_seq = transforms.Compose([
            random_params.IrfftToImg(shape, sd, decay_power, self.device)
        ])
        return transforms_seq

    def color_space_decorrelate(self,gamma, flag_decorr=None):
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

    def show_image(self, showed_img):
        # showed_img's size is 3*224*224, range is 0~1
        showed_img = np.transpose(showed_img, (1, 2, 0))
        plt.ioff()
        # plt.ion()
        fig = plt.figure(1, figsize=[2.24, 2.24], dpi=100, frameon=False)

        axis = plt.Axes(fig, [0., 0., 1., 1.])
        axis.set_axis_off()
        fig.add_axes(axis)
        axis.imshow(showed_img, interpolation='none')
        # # factorization_method = findall('[A-Z]', factorization_method)
        # factorization_method = ''.join(factorization_method)
        # plt.savefig(save_directory + '/' + attr_class + '_' + factorization_method + '_' +
        #             no_slash_layer_name + '_' + imgtype_name + str(index_saveimg) + '.jpeg')  # 'RdBu_r' 'hot'
        plt.show()
        print("")

    def transform_robustness(self, scale_param):
        transforms_seq = transforms.Compose([
            # transforms.Pad(12, padding_mode='reflect'),
            transforms.Pad(12),
            transform_robust.RandomCrop(8),

            transform_robust.RandomScale([n / 100. for n in range(scale_param[0], scale_param[1])]),
            transform_robust.RandomCrop(6),
            transform_robust.RandomRotate(list(range(-12, 12)) + list(range(-6, 6)) + 5 * list(range(-2, 2))),

            transform_robust.RandomCrop(2),
            transform_robust.RandomCrop(1),
            transform_robust.CropOrPadTo(self.image_height)
        ])
        return transforms_seq

    def gen_composed_img(self, t_rgb, t_alpha):
        t_bg = random_params.rand_fft_image(t_rgb.shape, sd=0.2, decay_power=1.5, device=self.device)
        t_bg = self.color_space_decorrelate(gamma=False)(t_bg)
        t_composed = t_bg * (1.0 - t_alpha) + t_rgb * t_alpha
        return t_composed, t_alpha

    def visualize_layer_with_hooks(self, flag_debug=False, flag_save=False,
                                   iteration_steps=501, flag_color_space=True):
        # Hook the selected layer
        self.hook_layer(self.batch_size, self.method_vis_neruon, self.selected_filters)
        shape = (self.batch_size, 3, self.image_height, self.image_height)

        # with decorr
        sd, decay_power = 0.01, 1.5

        gamma = False
        flag_decorr = 'KLT_MEAN'

        # scale_param = [85, 115]

        # for resnet 152 layer4
        scale_param = [105, 125]

        spectrums = random_params.rand_spectrum(shape, sd=sd, device=self.device)

        # Define optimizer for the image
        if self.opt_lr is None:
            optimizer = getattr(torch.optim, self.optimizer_name)([spectrums])
        else:
            # self.opt_lr = torch.tensor(self.opt_lr, dtype=torch.float32, device=self.device)
            optimizer = getattr(torch.optim, self.optimizer_name)([spectrums], lr=self.opt_lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=30,
                                                               threshold=0.009, min_lr=0.0005)

        save_step = int(iteration_steps / 1) - 1

        save_dir = self.gen_dir
        generic_im_path = save_dir + \
            '/vis' + \
            '_' + self.layer_name + '_' + str(self.layer_index) + \
            '_' + 'channel'

        for_best_loss = torch.tensor(1e7, dtype=torch.float32, device=self.device)

        for i_iter in range(1, iteration_steps):
            if i_iter % 500 == 0:
                flag_my_pace = False
            else:
                flag_my_pace = False

            if i_iter % 30 == 0:
                flag_my_infopace = True
            else:
                flag_my_infopace = False

            optimizer.zero_grad()

            # Assign create image to a variable to move forward in the model
            vis_image_no_rgb = self.process_to_img(shape, sd, decay_power)(spectrums)
            if flag_color_space:
                vis_image = torch.clamp(self.color_space_decorrelate(
                    gamma=gamma, flag_decorr=flag_decorr)(vis_image_no_rgb[:, 0:3, :, :]), 0, 1)
            else:
                vis_image = torch.clamp(vis_image_no_rgb[:, 0:3, :, :], 0, 1)
            # if flag_my_infopace:
            #     print('Iteration:', str(i_iter))
            #     print(vis_image.max())
            #     print(vis_image.min())
            x_rgb = self.transform_robustness(scale_param)(vis_image)
            x_rgb = self.pre_process_t()(x_rgb)

            output = self.model(x_rgb)

            loss_list = list(-torch.mean(self.conv_outputs[i_subloss]) for i_subloss in range(self.batch_size))

            loss = sum(loss_list)
            # Backward
            loss.backward()
            optimizer.step()

            scheduler.step(loss)
            if loss < for_best_loss or i_iter == 0:
                for_best_loss = loss
                vis_image_save = vis_image.clone().detach()

            # Save image
            if (flag_save and flag_my_pace) or (flag_save and i_iter == (iteration_steps - 1)):
                for i_savefig in range(self.batch_size):
                    im_path = generic_im_path + str(self.selected_filters[i_savefig]) + '.jpg'
                    vis_rgb_save = vis_image_save[i_savefig]
                    misc_functions.save_OI_PILImage_float(vis_rgb_save, im_path)

            # Show train information
            if flag_my_infopace:
                print('Iteration:', str(i_iter),
                      'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()),
                      'LR: {:.5f}'.format(optimizer.param_groups[0]['lr']))

