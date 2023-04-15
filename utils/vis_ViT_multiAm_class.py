import os
import numpy as np

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

from utils import transform_robust
from utils_params import random_params

from utils import misc_functions

from PIL import Image
from utils.util_loss import InversionMultiLoss

from utils.IO import get_proj_abs_dir


class ViTAMVisualization:
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, input_img_tensor, image_height,
                 layer_name='blocks', layer_index=4,
                 flag_layer='mid',
                 gen_dir=None,
                 optimizer_name='RMSprop', opt_lr=None,
                 loss_name='allRepr', loss_patch_index=1,
                 img_name=None, network_name='ViT_B_16',
                 flag_transparent=False):
        self.abs_dir = get_proj_abs_dir()
        self.model = model
        self.model.eval()

        self.input_img_tensor = input_img_tensor

        self.layer_name = layer_name
        self.layer_index = layer_index
        self.flag_layer = flag_layer

        self.image_height = image_height

        self.optimizer_name = optimizer_name
        self.opt_lr = opt_lr

        self.loss_name = loss_name
        self.loss_patch_index = loss_patch_index

        self.flag_transparent = flag_transparent

        gen_dir = gen_dir + '/' + network_name + '/img_info/' + img_name + '/' + 'visual_results_multiAm'
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

        self.kernel_small = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).to(self.device)

    def hook_layer(self):
        layer_module = getattr(self.model, self.layer_name)
        # for name, module in layer_module.named_modules():
        #     print(name)
        if self.flag_layer == 'mid':
            def hook_function_in1(module, am_in, am_out):
                self.conv_output1 = am_out[0]

            def hook_function_in2(module, am_in, am_out):
                self.conv_output2 = am_out[0]

            layer_module[self.layer_index - 1].register_forward_hook(hook_function_in1)
            layer_module[self.layer_index].attn.register_forward_hook(hook_function_in2)

        elif self.flag_layer == 'last':
            def hook_function_in(module, am_in, am_out):
                self.conv_output1 = am_in[0]

            def hook_function_out(module, am_in, am_out):
                self.conv_output2 = am_out
            # layer_module[self.layer_index].mlp.fc2.register_forward_hook(hook_function)
            # layer_module[self.layer_index].drop_path.register_forward_hook(hook_function)
            layer_module[self.layer_index].norm2.register_forward_hook(hook_function_in)
            layer_module[self.layer_index].mlp.drop2.register_forward_hook(hook_function_out)

        elif self.flag_layer == 'expec':
            def hook_function_out(module, am_in, am_out):
                self.conv_output1 = am_out[0]

            def hook_function_norm2out(module, am_in, am_out):
                self.conv_output3 = am_out

            layer_module[self.layer_index].register_forward_hook(hook_function_out)
            layer_module[self.layer_index].norm2.register_forward_hook(hook_function_norm2out)

        elif self.flag_layer == 'beforeTwoSum':
            def hook_function_out(module, am_in, am_out):
                self.conv_output1 = am_out[0]

            def hook_function_norm1out(module, am_in, am_out):
                self.conv_output2 = am_out

            def hook_function_norm2in(module, am_in, am_out):
                self.conv_output3 = am_in[0]

            layer_module[self.layer_index].register_forward_hook(hook_function_out)
            # layer_module[self.layer_index].norm1.register_forward_hook(hook_function_norm1out)
            layer_module[self.layer_index].norm2.register_forward_hook(hook_function_norm2in)

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
            beta: can be 1 or 2 or bigger
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
        if self.layer_index > 7:
            transforms_seq = transforms.Compose([
                transforms.Pad(12, padding_mode='reflect'),
                transform_robust.Closing(device=self.device),
                transform_robust.Opening(device=self.device),
                transform_robust.RandomRotate(5 * list(range(-2, 2))),

                transform_robust.RandomCrop(4),
                transform_robust.RandomCrop(1),
                transform_robust.CropOrPadTo(self.image_height)
            ])
        else:
            transforms_seq = transforms.Compose([
                transforms.Pad(8, padding_mode='reflect'),
                transform_robust.Closing(device=self.device),
                transform_robust.Opening(device=self.device),

                transform_robust.RandomRotate(5 * list(range(-2, 2))),

                transform_robust.RandomCrop(2),
                transform_robust.RandomCrop(1),
                transform_robust.CropOrPadTo(self.image_height)
            ])
        return transforms_seq

    def transform_robustness_latter(self):
        if self.layer_index > 1:
            transforms_seq = transforms.Compose([
                transform_robust.RandomCrop(1),
                transform_robust.CropOrPadTo(self.image_height)
            ])
        else:
            transforms_seq = transforms.Compose([
                # transform_robust.RandomCrop(1),
                transform_robust.CropOrPadTo(self.image_height)
            ])
        return transforms_seq

    def gen_composed_img(self, t_rgb, t_alpha, flag_decorr):
        t_bg = random_params.rand_fft_image(t_rgb.shape, sd=0.2, decay_power=1.5, device=self.device)
        t_bg = self.color_space_decorrelate(gamma=False, flag_decorr=flag_decorr)(t_bg)
        t_composed = t_bg * (1.0 - t_alpha) + t_rgb * t_alpha
        return t_composed, t_alpha

    def visualize_fm_with_hooks(self, flag_debug=False, flag_save=False,
                                flag_our_reg=True, flag_classical_reg=False, flag_color_space=True,
                                flag_save_loss=False,
                                iteration_steps=501):
        # Hook the selected layer
        self.hook_layer()
        shape = (self.batch_size, 3, self.image_height, self.image_height)
        sd, decay_power = 0.01, 1.9

        save_dir = self.gen_dir
        # save_dir = save_dir + '/sd{}_power{}'.format(sd, decay_power)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        gamma = False
        flag_decorr = 'KLT'  # 'KLT'   'KLT_MEAN'
        spectrums = random_params.rand_spectrum(shape, sd=sd, device=self.device)
        if self.opt_lr is None:
            optimizer = getattr(torch.optim, self.optimizer_name)([spectrums])
        else:
            # self.opt_lr = torch.tensor(self.opt_lr, dtype=torch.float32, device=self.device)
            optimizer = getattr(torch.optim, self.optimizer_name)([spectrums], lr=self.opt_lr,  betas=(0.9, 0.999))
            # optimizer = getattr(torch.optim, self.optimizer_name)([spectrums], lr=self.opt_lr, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=45,
                                                               threshold=0.009, min_lr=0.0001)
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
            # print("class is {}, score is {:.4f}".format(wordnet_ids[top5_catid[i]], top5_prob[i].item()))

        # ori_score = top5_prob[0].item()

        ori_ams_temp = self.conv_output1.detach().clone()
        # ori_norm1_temp = self.conv_output2.detach().clone()
        ori_norm2_temp = self.conv_output3.detach().clone()

        # ori_ams_temp_np = ori_ams_temp.detach().clone().cpu().numpy()
        # spatial_summed_am = np.sum(np.sum(ori_ams_temp_np[:, 1:, :], 0), 0)
        # spatial_maxed_am = np.max(np.max(ori_ams_temp_np[:, 1:, :], 0), 0)
        # spatial_mined_am = np.min(np.min(ori_ams_temp_np[:, 1:, :], 0), 0)

        attn_alpha = torch.tensor(1, dtype=torch.float32, device=self.device)
        for_best_loss = torch.tensor(10, dtype=torch.float32, device=self.device)
        # seventy_quantile = torch.quantile(ori_ams_temp, 0.75)
        # single_zero_tensor = torch.tensor(0., dtype=torch.float32, device=self.device)
        # ori_ams = torch.where(ori_ams_temp > seventy_quantile, ori_ams_temp, single_zero_tensor)

        save_step = int(iteration_steps / 100) - 1

        inversion_loss = InversionMultiLoss(self.loss_name, self.loss_patch_index)

        # alpha_tv = torch.tensor(2e-6, dtype=torch.float32, device=self.device)
        # alpha_reg_lambda = torch.tensor(2e-7, dtype=torch.float32, device=self.device)

        alpha_tv = torch.tensor(2e-5, dtype=torch.float32, device=self.device)
        alpha_reg_lambda = torch.tensor(2e-6, dtype=torch.float32, device=self.device)
        generic_im_path = save_dir + \
            '/vis' + \
            '_' + self.layer_name + str(self.layer_index) + \
            '_' + self.optimizer_name + \
            '_' + self.flag_layer + \
            '_' + self.loss_name

        if self.loss_name in ['singlePatch', 'singleChannel']:
            generic_im_path = generic_im_path + str(self.loss_patch_index)

        three_fourths_steps = int(3/4*iteration_steps)
        loss_values = []

        for i_iter in range(1, iteration_steps):
            if i_iter % save_step == 0:
                flag_my_pace = False
            else:
                flag_my_pace = False

            if i_iter % 80 == 0:
                flag_my_infopace = True
            else:
                flag_my_infopace = False

            optimizer.zero_grad()

            vis_image_no_rgb = self.process_to_img(shape, sd, decay_power)(spectrums)
            if flag_color_space:
                vis_image = torch.clamp(self.color_space_decorrelate(gamma=gamma,
                                        flag_decorr=flag_decorr)(vis_image_no_rgb[:, 0:3, :, :]), 0, 1)
            else:
                vis_image = torch.clamp(TF.adjust_brightness(vis_image_no_rgb[:, 0:3, :, :], 0.6), 0, 1)
                # vis_image = torch.clamp(TF.adjust_brightness(vis_image_no_rgb[:, 0:3, :, :], 2), 0, 1)
                # vis_image = self.pre_process_t()(vis_image)

            if flag_our_reg:
                if i_iter > three_fourths_steps:
                    x_rgb = self.transform_robustness_latter()(vis_image)
                else:
                    x_rgb = self.transform_robustness()(vis_image)
            else:
                x_rgb = vis_image
                # x_rgb = self.pre_process_t()(x_rgb)

            output = self.model(x_rgb)
            loss_list = []
            for i_subloss in range(self.batch_size):
                euc_loss = 1e-1 * inversion_loss(attn_alpha, None, None,
                                                 self.conv_output1[i_subloss], ori_ams_temp[i_subloss],
                                                 self.conv_output3[i_subloss], ori_norm2_temp[i_subloss])
                # # Calculate alpha regularization
                reg_alpha = alpha_reg_lambda * self.alpha_norm(vis_image[i_subloss], 6)
                # Calculate total variation regularization
                reg_total_variation = alpha_tv * self.total_variation_norm(vis_image[i_subloss], 4)

                # TODO: add L0 gradient minimization in future
                # Image Smoothing via L0 Gradient Minimization
                if flag_our_reg and flag_classical_reg:
                    # loss_list.append(euc_loss + reg_alpha + reg_total_variation)
                    loss_list.append(euc_loss + reg_alpha + reg_total_variation)
                elif flag_our_reg and (not flag_classical_reg):
                    loss_list.append(euc_loss)
                else:
                    loss_list.append(euc_loss)
            loss = sum(loss_list)

            # Backward
            loss.backward()
            optimizer.step()

            scheduler.step(loss)

            if loss < for_best_loss:
                for_best_loss = loss
                vis_image_save = vis_image.clone().detach()

            # Save image
            if (flag_save and flag_my_pace) or (i_iter == (iteration_steps - 1)):
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

            if flag_save_loss:
                loss_value = loss.data.cpu().numpy()
                loss_values.append(loss_value)

            # Show train information
            if flag_my_infopace:
                loss_value = loss.data.cpu().numpy()
                if len(loss_value.shape) > 0:
                    loss_value = loss_value[0]
                print('Iteration:', str(i_iter),
                      'Loss:', "{0:.4f}".format(loss_value),
                      'LR: {:.9f}'.format(optimizer.param_groups[0]['lr']))

        if flag_save_loss:
            loss_path = save_dir +\
                        '/loss' + \
                        '_' + self.layer_name + str(self.layer_index) + \
                        '_' + self.optimizer_name + \
                        '_' + str(iteration_steps) + '.txt'
            with open(loss_path, "w") as f:
                for loss_value in loss_values:
                    # write each item on a new line
                    f.write("{:.4f}\n".format(loss_value))
                print('Done')
