import torch
from torch import Tensor


class InversionLoss(torch.nn.Module):
    def __init__(self, loss_name='allRepr', patch_index=1) -> None:
        super(InversionLoss, self).__init__()
        self.loss_name = loss_name
        self.patch_index = patch_index

    def forward(self, am, attn, target_am, target_attn, attn_alpha):
        if self.loss_name == 'allRepr':
            am_term = torch.square(torch.norm(am[:] - target_am[:])) /\
                      torch.square(torch.norm(target_am[:]))
            attn_term = 0.0
        elif self.loss_name == 'multiAllRepr':
            am_term = torch.square(torch.norm(am[:] - target_am[:])) /\
                      torch.square(torch.norm(target_am[:]))
            attn_term = torch.square(torch.norm(attn[:] - target_attn[:])) /\
                        torch.square(torch.norm(target_attn[:]))
        elif self.loss_name == '196Patch':
            am_term = torch.square(torch.norm(am[1:] - target_am[1:])) /\
                      torch.square(torch.norm(target_am[1:]))
            attn_term = 0.0
        elif self.loss_name == 'class1Patch':
            am_term = torch.square(torch.norm(am[0] - target_am[0])) / \
                      torch.square(torch.norm(target_am[0]))
            attn_term = 0.0
        elif self.loss_name == 'singlePatch':
            am_term = torch.square(torch.norm(am[self.patch_index] - target_am[self.patch_index])) /\
                      torch.square(torch.norm(target_am[self.patch_index]))
            attn_term = 0.0
        elif self.loss_name == 'singleChannel':
            am_term = torch.square(torch.norm(am[:, self.patch_index] - target_am[:, self.patch_index])) /\
                      torch.square(torch.norm(target_am[:, self.patch_index]))
            attn_term = 0.0
        elif self.loss_name == 'reprAttn':
            am_term = torch.square(torch.norm(am - target_am)) /\
                      torch.square(torch.norm(target_am))
            attn_term = torch.square(torch.norm(attn[:, 0, :] - target_attn[:, 0, :])) / \
                        torch.square(torch.norm(target_attn[:, 0, :]))
        elif self.loss_name == 'allAttn':
            am_term = 0.0
            attn_term = torch.square(torch.norm(attn - target_attn)) / \
                        torch.square(torch.norm(target_attn))
        elif self.loss_name == 'noClassAttn':
            am_term = 0.0
            attn_term = torch.square(torch.norm(attn[:, 1:, 1:] - target_attn[:, 1:, 1:])) / \
                        torch.square(torch.norm(target_attn[:, 1:, 1:]))

        return am_term + attn_term * attn_alpha


class InversionMultiLoss(torch.nn.Module):
    def __init__(self, loss_name='allRepr', patch_index=1) -> None:
        super(InversionMultiLoss, self).__init__()
        self.loss_name = loss_name
        self.patch_index = patch_index

    def forward(self, attn_alpha, attr_mask=None, norm2_mask=None, *args):
        # args 顺序是am target_am norm target_norm这样排列
        if 3 <= len(args) < 5:
            if attr_mask is not None:
                am_term = torch.square(torch.norm((args[0][:] - args[1][:])*attr_mask)) / \
                          torch.square(torch.norm(args[1][:]*attr_mask))
                norm2_term = torch.square(torch.norm((args[2][:] - args[3][:])*norm2_mask)) / \
                             torch.square(torch.norm(args[3][:]*norm2_mask))
                # am_term = torch.square(torch.norm(args[0][:] - args[1][:]*attr_mask)) / \
                #           torch.square(torch.norm(args[1][:]*attr_mask))
                # norm2_term = torch.square(torch.norm(args[2][:] - args[3][:]*norm2_mask)) / \
                #              torch.square(torch.norm(args[3][:]*norm2_mask))
            else:
                am_term = torch.square(torch.norm(args[0][:] - args[1][:])) / \
                          torch.square(torch.norm(args[1][:]))
                # norm1_term = torch.square(torch.norm(norm1[:] - target_norm1[:])) / \
                #              torch.square(torch.norm(target_norm1[:]))

                norm2_term = torch.square(torch.norm(args[2][:] - args[3][:])) / \
                             torch.square(torch.norm(args[3][:]))
            return am_term + norm2_term * attn_alpha
        elif len(args) < 3:
            am_term = torch.square(torch.norm(args[0][:] - args[1][:])) / \
                      torch.square(torch.norm(args[1][:]))
            return am_term
        else:
            am_term = torch.square(torch.norm((args[0][:] - args[1][:]))) / \
                      torch.square(torch.norm(args[1][:]))
            mean_term = torch.square(torch.norm((args[2][:] - args[3][:]))) / \
                             torch.square(torch.norm(args[3][:]))
            var_term = torch.square(torch.norm((args[4][:] - args[5][:]))) / \
                             torch.square(torch.norm(args[5][:]))
            # mean_term = torch.norm((args[2][:] - args[3][:]))
            # var_term = torch.norm((args[4][:] - args[5][:]))
            return am_term + mean_term * attn_alpha + var_term * attn_alpha


class MultiAMMaximizationLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(MultiAMMaximizationLoss, self).__init__()

    def forward(self, attn_alpha, attr_mask=None, norm2_mask=None, *args):
        am_term = torch.sum(args[0][:]*args[1][:]*attr_mask)
        if norm2_mask is not None:
            norm2_term = torch.sum(args[2][:]*args[3][:]*norm2_mask)
            return am_term + norm2_term * attn_alpha
        else:
            return am_term


class AMMaximizationLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(AMMaximizationLoss, self).__init__()

    def forward(self, attr_mask=None, *args):
        am_term = torch.sum(args[0][:]*args[1][:]*attr_mask)
        return am_term


class AttrMaximizationLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(AttrMaximizationLoss, self).__init__()

    def forward(self, vis_am, vis_norm2, ori_img_attr, ori_img_norm2, attn_alpha):
        am_term = torch.sum(vis_am[:] * ori_img_attr[:])
        norm2_term = torch.sum(vis_norm2[:] * ori_img_norm2[:])

        # am_term = torch.square(torch.norm(vis_am[1:, ] - ori_img_attr)) / \
        #           torch.square(torch.norm(ori_img_attr))
        #
        # norm2_term = torch.square(torch.norm(vis_norm2[:] - ori_img_norm2[:])) / \
        #              torch.square(torch.norm(ori_img_norm2[:]))

        return am_term + norm2_term * attn_alpha


class AttrEucLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(AttrEucLoss, self).__init__()

    def forward(self, vis_am, vis_norm2, ori_img_attr, ori_img_norm2, attn_alpha):
        am_term = torch.square(torch.norm(vis_am[1:, :] - ori_img_attr[:])) / \
                  torch.square(torch.norm(ori_img_attr[:]))
        # norm1_term = torch.square(torch.norm(norm1[:] - target_norm1[:])) / \
        #              torch.square(torch.norm(target_norm1[:]))

        norm2_term = torch.square(torch.norm(vis_norm2[1:, :] - ori_img_norm2[:])) / \
                     torch.square(torch.norm(ori_img_norm2[:]))

        return am_term + norm2_term * attn_alpha
