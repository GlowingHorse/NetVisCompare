import torch
import torchvision.transforms.functional as TF
import random
from torchvision import transforms
import numpy as np
import PIL
import kornia
from kornia import morphology as morph


def _rand_select(xs):
    xs_list = list(xs)
    rand_n = np.random.randint(0, len(xs_list), 1)
    return xs_list[rand_n[0]]


class MedianFilter(torch.nn.Module):
    def __init__(self, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_sizes = [3, 5, 7, 9]
            random_idx = np.random.randint(0, 4, 1)[0]
            self.kernel_size = kernel_sizes[random_idx]
        else:
            self.kernel_size = kernel_size

    def forward(self, img):
        # t_shp = img.size()
        opened_image = kornia.filters.median_blur(img, (self.kernel_size, self.kernel_size))
        return opened_image


class MeanFilter(torch.nn.Module):
    def __init__(self, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_sizes = [3, 5, 7, 9]
            random_idx = np.random.randint(0, 4, 1)[0]
            self.kernel_size = kernel_sizes[random_idx]
        else:
            self.kernel_size = kernel_size

    def forward(self, img):
        # t_shp = img.size()
        opened_image = kornia.filters.box_blur(img, (self.kernel_size, self.kernel_size))
        return opened_image


class Opening(torch.nn.Module):
    def __init__(self, kernel=None, device='cpu'):
        super().__init__()
        if kernel is None:
            kernel0 = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).to(device)
            kernel1 = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 0]]).to(device)
            kernel2 = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 1, 0]]).to(device)
            kernel3 = torch.tensor([[0, 1, 0], [1, 1, 1], [1, 1, 0]]).to(device)
            kernel4 = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 1]]).to(device)
            kernels = [kernel0, kernel1, kernel2, kernel3, kernel4]
            rand_n = np.random.randint(0, 5, 1)
            kernel = kernels[rand_n[0]]
            # # middle kernel
            # kernel = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).to(device)

            # # big kernel
            # kernel = torch.tensor(
            #     [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]]).to(device)
        self.kernel = kernel

    def forward(self, img):
        # t_shp = img.size()
        opened_image = morph.opening(img, self.kernel)
        return opened_image


class Closing(torch.nn.Module):
    def __init__(self, kernel=None, device='cpu'):
        super().__init__()
        if kernel is None:
            kernel0 = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).to(device)
            kernel1 = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 0]]).to(device)
            kernel2 = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 1, 0]]).to(device)
            kernel3 = torch.tensor([[0, 1, 0], [1, 1, 1], [1, 1, 0]]).to(device)
            kernel4 = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 1]]).to(device)

            # kernel0 = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).to(device)
            # kernel1 = torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]]).to(device)
            # kernel2 = torch.tensor([[0, 1, 1], [0, 1, 1], [0, 0, 0]]).to(device)
            # kernel3 = torch.tensor([[0, 0, 0], [1, 1, 0], [1, 1, 0]]).to(device)
            # kernel4 = torch.tensor([[0, 0, 0], [0, 1, 1], [0, 1, 1]]).to(device)
            kernels = [kernel0, kernel1, kernel2, kernel3, kernel4]
            rand_n = np.random.randint(0, 5, 1)
            kernel = kernels[rand_n[0]]
            # # middle kernel
            # kernel = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).to(device)

            # # big kernel
            # kernel = torch.tensor(
            #     [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]]).to(device)
        self.kernel = kernel

    def forward(self, img):
        # t_shp = img.size()
        opened_image = morph.closing(img, self.kernel)
        return opened_image


class RandomCrop(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, img):
        t_shp = img.size()
        img = transforms.RandomCrop(t_shp[-1]-self.d)(img)
        return img

    # def __repr__(self):
    #     return self.__class__.__name__ + '(jittering={0})'.\
    #         format(self.d)


class CropOrPadTo(torch.nn.Module):
    def __init__(self, height):
        super().__init__()
        self.height = height

    def forward(self, img):
        t_shp = img.size()
        if t_shp[-1] < self.height:
            padding_size = int((self.height - t_shp[-1])/2)
            img = transforms.Pad(padding_size, padding_mode='reflect')(img)
        else:
            img = transforms.CenterCrop(self.height)(img)
        img = transforms.Resize(self.height)(img)
        return img

    # def __repr__(self):
    #     return self.__class__.__name__ + '(croporpadto={0})'.\
    #         format(self.height)


class RandomScale(torch.nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.scales = scales

    def forward(self, img):
        t_shp = img.size()
        scale = _rand_select(self.scales)
        # img = transforms.Resize(int(t_shp[-1] * scale), PIL.Image.BILINEAR)(img)
        img = transforms.Resize(int(t_shp[-1] * scale), TF.InterpolationMode.BILINEAR)(img)
        return img


class RandomRotate(torch.nn.Module):
    def __init__(self, angles):
        super().__init__()
        self.angles = angles

    def forward(self, img):
        angle = _rand_select(self.angles)
        img = TF.rotate(img, angle, TF.InterpolationMode.BILINEAR)
        return img


# _KLT color space decompose
# a random example is
mean_KLT_decorr_matrix = np.asarray([[0.430,  0.411,  0.188],
                                     [0.479, -0.031, -0.168],
                                     [0.519,  0.297, -0.062]]).astype("float32")
ortho_KLT_decorr_matrix = np.asarray([[0.430,  0.30039364, -0.063219],
                                      [0.411, -0.20171449,  0.162292],
                                      [0.188, -0.24608836, -0.210199]]).astype("float32")


def _KLT(t, device='cpu'):
    permuted_t = t.permute(1, 0, 2, 3)
    permuted_t_shp = permuted_t.size()
    t_flat = torch.reshape(permuted_t, [3, -1])
    # _, eigenVec = torch.linalg.eig(torch.cov(t_flat))
    # # t = torch.matmul(eigenVec.real.T, t_flat -
    # #                  torch.unsqueeze(torch.tensor([0.485, 0.456, 0.406], device=device), 1))
    # t = torch.matmul(torch.linalg.inv(eigenVec.real.T), t_flat -
    #                  torch.unsqueeze(torch.tensor([0.485, 0.456, 0.406], device=device), 1))
    # # t = torch.matmul(eigenVec.real.T, t_flat)

    normed_KLTdecomp_sqrt = torch.from_numpy(ortho_KLT_decorr_matrix).float()
    normed_KLTdecomp_sqrt = normed_KLTdecomp_sqrt.to(device=device)
    t = torch.matmul(normed_KLTdecomp_sqrt, t_flat)

    t = torch.reshape(t, permuted_t_shp)
    t = t.permute(1, 0, 2, 3)
    return t


def _KLT_MEAN(t, device='cpu'):
    permuted_t = t.permute(1, 0, 2, 3)
    permuted_t_shp = permuted_t.size()
    t_flat = torch.reshape(permuted_t, [3, -1])
    # _, eigenVec = torch.linalg.eig(torch.cov(t_flat))
    # # t = torch.matmul(eigenVec.real.T, t_flat -
    # #                  torch.unsqueeze(torch.tensor([0.485, 0.456, 0.406], device=device), 1))
    # t = torch.matmul(torch.linalg.inv(eigenVec.real.T), t_flat -
    #                  torch.unsqueeze(torch.tensor([0.485, 0.456, 0.406], device=device), 1))
    # # t = torch.matmul(eigenVec.real.T, t_flat)

    normed_KLTdecomp_sqrt = torch.from_numpy(ortho_KLT_decorr_matrix).float()
    normed_KLTdecomp_sqrt = normed_KLTdecomp_sqrt.to(device=device)
    t = torch.matmul(normed_KLTdecomp_sqrt, t_flat)
    img_mean_tensor = torch.tensor([[0.485], [0.456], [0.406]], dtype=torch.float32, device=device)
    t = t + img_mean_tensor
    t = torch.reshape(t, permuted_t_shp)
    t = t.permute(1, 0, 2, 3)
    return t


# Don't seem to improve vis quality
def _I1I2I3(t, device='cpu'):
    permuted_t = t.permute(1, 0, 2, 3)
    permuted_t_shp = permuted_t.size()
    t_flat = torch.reshape(permuted_t, [3, -1])
    color_corr = torch.from_numpy(np.asarray([[0.33, 0.33, 0.33],
                                              [0.5, 0.00, -0.5],
                                              [-0.25, 0.5, -0.25]]).astype("float32")).float()
    color_corr = color_corr.to(device=device)
    t_flat = torch.matmul(color_corr, t_flat)
    t = torch.reshape(t_flat, permuted_t_shp)
    t = t.permute(1, 0, 2, 3)
    return t


def color_decorrelate(t, gamma=False, flag_decorr=None, device='cpu'):
    if flag_decorr is not None:
        if flag_decorr == 'KLT':
            t = _KLT(t, device=device)
        elif flag_decorr == 'I1I2I3':
            t = _I1I2I3(t, device=device)
        elif flag_decorr == 'KLT_MEAN':
            t = _KLT_MEAN(t, device=device)

    t = torch.sigmoid(t)
    if gamma:
        t = TF.adjust_brightness(t, 1.35)
        # t = TF.adjust_brightness(t, 0.65)
    return t


class ColorDecorrelation(torch.nn.Module):
    def __init__(self, gamma, device, flag_decorr=None):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.flag_decorr = flag_decorr

    def forward(self, img):
        img = color_decorrelate(img, gamma=self.gamma,
                           flag_decorr=self.flag_decorr, device=self.device)
        return img


class ClipColorDecorrelation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        img = torch.clamp(img, 0, 1)
        return img

