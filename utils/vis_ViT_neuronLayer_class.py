import os

import torch
from torchvision import transforms

from utils import transform_robust
from utils_params import random_params

import matplotlib
matplotlib.use('Agg')
from utils import misc_functions


class ViTLayerVisualization:
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model,
                 selected_filters=None, image_height=224,
                 layer_name='blocks', layer_index=4,
                 gen_dir=None,
                 optimizer_name='RMSprop', opt_lr=None,
                 loss_name='positive',
                 network_name='ViT_B_16',
                 method_vis_neruon=None,
                 vis_mode='non_certain_img'):
        self.model = model
        self.model.eval()

        # self.selected_filters = selected_filters

        self.layer_name = layer_name
        self.layer_index = layer_index

        self.image_height = image_height

        self.optimizer_name = optimizer_name
        self.opt_lr = opt_lr

        self.loss_name = loss_name

        self.method_vis_neruon = method_vis_neruon

        if self.method_vis_neruon == 'channel-level' and vis_mode == 'non_certain_img':
            self.gen_dir = gen_dir + '/' + network_name + '/' + \
                      'channel-level-vis' + '/' + \
                      layer_name + str(layer_index)
        elif self.method_vis_neruon == 'channel-level' and vis_mode == 'certain_img':
            self.gen_dir = gen_dir + '/' + network_name + '/' + \
                      'channel-level-vis4certain-img' + '/' + \
                      layer_name + str(layer_index)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            self.model.to(self.device)

        # Create the folder to export images if not exists
        if not os.path.exists(self.gen_dir):
            os.makedirs(self.gen_dir)

        self.selected_filters = selected_filters
        self.batch_size = len(self.selected_filters)

    def hook_layer(self):
        def hook_function(module, am_in, am_out):
            self.conv_outputs = []
            for i_batch_size in range(am_out[0].shape[0]):
                self.conv_outputs.append(am_out[0][i_batch_size, 1:, :])
        layer_module = getattr(self.model, self.layer_name)
        layer_module[self.layer_index].register_forward_hook(hook_function)

    def alpha_norm(self, input_matrix, alpha):
        """
            Converts matrix to vector then calculates the alpha norm
        """
        alpha_norm = ((input_matrix.contiguous().view(-1))**alpha).sum()
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

    def transform_robustness(self, scale_param):
        if self.layer_index > 1:
            transforms_seq = transforms.Compose([
                transforms.Pad(12, padding_mode='reflect'),
                transform_robust.RandomCrop(8),
                transform_robust.RandomScale([n / 100. for n in range(scale_param[0], scale_param[1])]),

                transform_robust.Closing(device=self.device),
                transform_robust.Opening(device=self.device),

                transform_robust.RandomRotate(list(range(-12, 12)) + list(range(-6, 6))
                                              + 5 * list(range(-2, 2))),

                transform_robust.Closing(device=self.device),
                transform_robust.Opening(device=self.device),

                transform_robust.RandomCrop(2),
                # transform_robust.Closing(kernel_small, self.device),
                transform_robust.CropOrPadTo(self.image_height),
                # transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1)
            ])
        else:
            transforms_seq = transforms.Compose([
                transforms.Pad(10, padding_mode='reflect'),

                # transform_robust.Closing(kernel_small, self.device),
                transform_robust.RandomCrop(6),
                transform_robust.RandomScale([n / 100. for n in range(85, 100)]),

                transform_robust.Closing(device=self.device),
                transform_robust.Opening(device=self.device),

                transform_robust.RandomRotate(5 * list(range(-2, 2))),

                transform_robust.Closing(device=self.device),
                transform_robust.Opening(device=self.device),

                transform_robust.RandomCrop(2),
                transform_robust.RandomCrop(1),
                transform_robust.CropOrPadTo(self.image_height),
                # transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1)
            ])
        return transforms_seq

    def gen_composed_img(self, t_rgb, t_alpha):
        t_bg = random_params.rand_fft_image(t_rgb.shape, sd=0.2, decay_power=1.5, device=self.device)
        t_bg = self.color_space_decorrelate(gamma=False)(t_bg)
        t_composed = t_bg * (1.0 - t_alpha) + t_rgb * t_alpha
        return t_composed, t_alpha

    def visualize_layer_with_hooks(self, flag_save=False,
                                   flag_color_space=True,
                                   iteration_steps=501):
        # Hook the selected layer
        self.hook_layer()
        shape = (self.batch_size, 3, self.image_height, self.image_height)
        sd, decay_power = 0.01, 1.5
        scale_param = [80, 120]

        gamma = False
        flag_decorr = 'KLT_MEAN'

        spectrums = random_params.rand_spectrum(shape, sd=sd, device=self.device)

        if self.opt_lr is None:
            optimizer = getattr(torch.optim, self.optimizer_name)([spectrums])
        else:
            optimizer = getattr(torch.optim, self.optimizer_name)([spectrums], lr=self.opt_lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=45,
                                                               threshold=0.009, min_lr=0.003)

        save_step = int(iteration_steps / 1) - 1
        save_dir = self.gen_dir
        generic_im_path = save_dir + \
            '/vis' + \
            '_' + self.layer_name + str(self.layer_index) + \
            '_' + 'channel'

        if self.method_vis_neruon in ['channel-neuron-level']:
            png_save_dir = save_dir + '/png_imgs'
            if not os.path.exists(png_save_dir):
                os.makedirs(png_save_dir)
            generic_png_im_path = png_save_dir + \
                '/vis' + \
                '_' + self.layer_name + str(self.layer_index) + \
                '_' + 'channel'

        for_best_loss = torch.tensor(1e6, dtype=torch.float32)
        alpha_tv = torch.tensor(2e-6, dtype=torch.float32, device=self.device)
        alpha_reg_lambda = torch.tensor(2e-7, dtype=torch.float32, device=self.device)

        for i_iter in range(1, iteration_steps):
            if i_iter % save_step == 0:
                flag_my_pace = False
            else:
                flag_my_pace = False

            if i_iter % 40 == 0:
                flag_my_infopace = True
            else:
                flag_my_infopace = False

            optimizer.zero_grad()

            vis_image_no_rgb = self.process_to_img(shape, sd, decay_power)(spectrums)
            if flag_color_space:
                vis_image = torch.clamp(self.color_space_decorrelate(gamma=gamma,
                                        flag_decorr=flag_decorr)(vis_image_no_rgb[:, 0:3, :, :]), 0, 1)
            else:
                vis_image = torch.clamp(vis_image_no_rgb[:, 0:3, :, :], 0, 1)

            x_rgb = self.transform_robustness(scale_param)(vis_image)
            x_rgb = self.pre_process_t()(x_rgb)

            output = self.model(x_rgb)
            loss_list = []
            for i_subloss in range(self.batch_size):
                # # Calculate alpha regularization
                reg_alpha = alpha_reg_lambda * self.alpha_norm(vis_image[i_subloss], 6)
                # Calculate total variation regularization
                reg_total_variation = alpha_tv * self.total_variation_norm(vis_image[i_subloss], 2)

                unselect_conv_output_temp = self.conv_outputs[i_subloss]
                select_conv_output = unselect_conv_output_temp[:, self.selected_filters[i_subloss]]

                if self.loss_name == 'positive':
                    actMax_loss = -torch.mean(select_conv_output)
                    loss_list.append(actMax_loss)
                elif self.loss_name == 'negative':
                    actMin_loss = torch.mean(select_conv_output)
                    loss_list.append(actMin_loss)
                elif isinstance(self.loss_name, int):
                    ref_activation = torch.tensor(self.loss_name, dtype=torch.float32, device=self.device)
                    refAct_loss = 1e-1 * torch.square(torch.norm(select_conv_output - ref_activation)) / \
                                  torch.square(torch.norm(ref_activation))
                    loss_list.append(refAct_loss + reg_alpha + reg_total_variation)

            # Backward
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()

            scheduler.step(loss)

            if loss < for_best_loss or i_iter == 0:
                for_best_loss = loss
                vis_image_save = vis_image.clone().detach()

            # Save image
            if (flag_save and flag_my_pace) or (i_iter == (iteration_steps - 1)):
                for i_savefig in range(self.batch_size):
                    im_path = generic_im_path + str(self.selected_filters[i_savefig]) + \
                              '_' + 'loss_' + str(self.loss_name) + '.jpg'
                    vis_rgb_save = vis_image_save[i_savefig]

                    # misc_functions.save_OI_PILImage_float(vis_rgb_save, im_path)
                    misc_functions.save_OI_PILImage_float(vis_rgb_save, im_path)

            # Show train information
            if flag_my_infopace:
                loss_value = loss.data.cpu().numpy()
                if len(loss_value.shape) > 0:
                    loss_value = loss_value[0]
                print('Iteration:', str(i_iter),
                      'Loss:', "{0:.4f}".format(loss_value),
                      'LR: {:.9f}'.format(optimizer.param_groups[0]['lr']))

