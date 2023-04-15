import os
import copy
import numpy as np

import torch
import torch.fft
from torch.autograd import Variable
from torchvision import models
from utils import transform_robust

from matplotlib import pyplot as plt
from pytorch_wavelets import DTCWTForward, DTCWTInverse


def rand_wavelets_params(shape, biort, qshift, device='cpu'):
    b, ch, h, w = shape

    # Yh[0] - max: 0.2048, min: -0.1691, std: 0.0103, mid: -0.0000, mean: -0.0000, size: (1, 3, 6, 112, 112, 2)
    # Yh[1] - max: 0.5503, min: -0.5572, std: 0.0356, mid: 0.0001, mean: 0.0000, size: (1, 3, 6, 56, 56, 2)
    # Yh[2] - max: 1.1430, min: -0.9118, std: 0.1022, mid: 0.0001, mean: 0.0002, size: (1, 3, 6, 28, 28, 2)
    # Yl - max: 3.9509, min: -0.0718, std: 0.8448, mid: 1.5287, mean: 1.6572, size: (1, 3, 56, 56)

    # ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').cuda()

    # Yh0 = np.random.normal(0, 0.0103, (b, ch, 6, int(h/2), int(h/2), 2))
    # Yh1 = np.random.normal(0, 0.0356, (b, ch, 6, int(h/4), int(h/4), 2))
    # Yh2 = np.random.normal(0, 0.1022, (b, ch, 6, int(h/8), int(h/8), 2))
    # Yl = np.random.normal(1.6572, 0.8448, (b, ch, int(h/4), int(h/4)))
    Yh0 = np.load('../data/wavelet_init/' + biort + '_' + qshift + 'Yh0.npy')
    Yh1 = np.load('../data/wavelet_init/' + biort + '_' + qshift + 'Yh1.npy')
    Yh2 = np.load('../data/wavelet_init/' + biort + '_' + qshift + 'Yh2.npy')
    Yl = np.load('../data/wavelet_init/' + biort + '_' + qshift + 'Yl.npy')

    Yh0_tensor = torch.from_numpy(Yh0).float().to(device)
    Yh1_tensor = torch.from_numpy(Yh1).float().to(device)
    Yh2_tensor = torch.from_numpy(Yh2).float().to(device)
    Yl_tensor = torch.from_numpy(Yl).float().to(device)

    Yh0_var = Variable(Yh0_tensor, requires_grad=True)
    Yh1_var = Variable(Yh1_tensor, requires_grad=True)
    Yh2_var = Variable(Yh2_tensor, requires_grad=True)
    Yl_var = Variable(Yl_tensor, requires_grad=True)
    return [Yh0_var, Yh1_var, Yh2_var, Yl_var]


class WaveletsParamToImg(torch.nn.Module):
    def __init__(self, shape, biort, qshift, device):
        super().__init__()
        self.device = device
        self.shape = shape
        # self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').cuda()
        # 'antonini', 'legall', 'near_sym_a', 'near_sym_b'
        # 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c', 'qshift_d'
        # self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').cuda()
        self.ifm = DTCWTInverse(biort=biort, qshift=qshift).cuda()

    def forward(self, params):
        Yh0_var, Yh1_var, Yh2_var, Yl_var = params
        imgs = self.ifm((Yl_var, [Yh0_var, Yh1_var, Yh2_var]))
        return imgs


def _rfft2d_freqs(h, w):
    """Compute 2d spectrum frequences."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[:w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[:w // 2 + 1]
    struc_2d_freq = np.sqrt(fx * fx + fy * fy)
    return struc_2d_freq


def rand_spectrum(shape, sd=None, device=None):
    b, ch, h, w = shape
    sd = 0.01 if sd is None else sd

    spec_init_vars = []
    for _ in range(b):
        freqs = _rfft2d_freqs(h, w)
        fh, fw = freqs.shape

        # init_val = sd * np.random.randn(2, ch, fh, fw).astype("float32")
        # init_val = torch.from_numpy(init_val).float()

        rand_init_val = sd * torch.randn(2, ch, fh, fw, dtype=torch.float32)
        spec_init_vars.append(rand_init_val)
    spec_init_vars = torch.stack(spec_init_vars, 0)
    spec_init_vars = spec_init_vars.to(device)
    spec_init_vars = Variable(spec_init_vars, requires_grad=True)
    return spec_init_vars


def rand_noise_img(shape, device):
    # init_random_noise_tensor = torch.normal(mean=0.5, std=0.5, size=shape)

    init_random_noise_tensor = 1e-1 * torch.randn(shape)
    init_random_noise_tensor = init_random_noise_tensor.to(device)
    init_random_noise_tensor = Variable(init_random_noise_tensor, requires_grad=True)
    return init_random_noise_tensor


class IrfftToImg(torch.nn.Module):
    def __init__(self, shape, sd, decay_power, device):
        super().__init__()
        self.device = device
        self.shape = shape
        self.sd = sd
        self.decay_power = decay_power

        b, ch, h, w = self.shape
        sampled_fft_freqs = _rfft2d_freqs(h, w)
        scale_factor = 1.0 / np.maximum(sampled_fft_freqs, 1.0 / max(h, w)) ** self.decay_power
        scale_factor *= np.sqrt(w * h)
        scale_factor = torch.from_numpy(scale_factor).float()

        self.scale_factor = scale_factor.to(self.device)

    def forward(self, spectrum_vars):
        batch_num, ch, h, w = self.shape
        imgs = []
        for i in range(batch_num):
            spectrum_var = spectrum_vars[i]
            spectrum = torch.complex(spectrum_var[0], spectrum_var[1])
            scaled_spectrum = spectrum * self.scale_factor
            # img = torch.fft.irfftn(scaled_spectrum)
            img = torch.fft.irfft2(scaled_spectrum)
            # in case of odd input dimension we cut off the additional pixel
            # we get from irfft2d length computation
            img = img[:ch, :h, :w]
            imgs.append(img)

        # TODO: check which range is best
        # return torch.stack(imgs) * 3.
        return torch.stack(imgs) / 4.
        # return torch.stack(imgs) / 6.
        # return torch.stack(imgs)


# ----------------------------------------------------------------------
# Use for generating Fourier noise image directly
def rand_fft_image(shape, sd=None, decay_power=1.1, device='cpu'):
    b, ch, h, w = shape
    sd = 0.01 if sd is None else sd

    imgs = []
    for _ in range(b):
        freqs = _rfft2d_freqs(h, w)
        fh, fw = freqs.shape
        init_val = sd * np.random.randn(2, ch, fh, fw).astype("float32")
        init_val = torch.from_numpy(init_val).float()
        init_val = init_val.to(device)
        spectrum_var = Variable(init_val, requires_grad=True)

        spectrum = torch.complex(spectrum_var[0], spectrum_var[1])
        # print(spectrum.is_leaf)

        scale_factor = 1.0 / np.maximum(freqs, 1.0 / max(h, w)) ** decay_power

        # plt.imshow(np.maximum(freqs, 1.0 / max(h, w)), interpolation='nearest')
        # plt.show()
        # plt.imshow(freqs, interpolation='nearest')
        # plt.show()

        # Scale the spectrum by the square-root of the number of pixels
        # to get a unitary transformation. This allows to use similar
        # leanring rates to pixel-wise optimisation.
        scale_factor *= np.sqrt(w * h)
        # plt.imshow(scale_factor, interpolation='nearest')
        # plt.show()
        scale_factor = torch.from_numpy(scale_factor).float()
        scale_factor = scale_factor.to(device)
        scaled_spectrum = spectrum * scale_factor
        # print(scaled_spectrum.is_leaf)
        # scaled_spectrum_value = scaled_spectrum.eval()
        # scaled_spectrum_value = np.absolute(scaled_spectrum_value)

        img = torch.fft.irfft2(scaled_spectrum)
        # in case of odd input dimension we cut off the additional pixel
        # we get from irfft2d length computation
        img = img[:ch, :h, :w]
        imgs.append(img)
    return torch.stack(imgs) / 4.
    # return torch.stack(imgs)


# ----------------------------------------------------------------------
# Use for testing generate wavelets noise image directly
def rand_wavelets_image(shape, device='cpu'):
    b, ch, h, w = shape

    # Yh[0] - max: 0.2048, min: -0.1691, std: 0.0103, mid: -0.0000, mean: -0.0000, size: (1, 3, 6, 112, 112, 2)
    # Yh[1] - max: 0.5503, min: -0.5572, std: 0.0356, mid: 0.0001, mean: 0.0000, size: (1, 3, 6, 56, 56, 2)
    # Yh[2] - max: 1.1430, min: -0.9118, std: 0.1022, mid: 0.0001, mean: 0.0002, size: (1, 3, 6, 28, 28, 2)
    # Yl - max: 3.9509, min: -0.0718, std: 0.8448, mid: 1.5287, mean: 1.6572, size: (1, 3, 56, 56)

    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').cuda()

    Yh0 = np.random.normal(0, 0.0103, (b, ch, 6, int(h/2), int(h/2), 2))
    Yh1 = np.random.normal(0, 0.0356, (b, ch, 6, int(h/4), int(h/4), 2))
    Yh2 = np.random.normal(0, 0.1022, (b, ch, 6, int(h/8), int(h/8), 2))
    Yl = np.random.normal(1.6572, 0.8448, (b, ch, int(h/4), int(h/4)))

    Yh0_tensor = torch.from_numpy(Yh0).float().to(device)
    Yh1_tensor = torch.from_numpy(Yh1).float().to(device)
    Yh2_tensor = torch.from_numpy(Yh2).float().to(device)
    Yl_tensor = torch.from_numpy(Yl).float().to(device)

    Yh0_var = Variable(Yh0_tensor, requires_grad=True)
    Yh1_var = Variable(Yh1_tensor, requires_grad=True)
    Yh2_var = Variable(Yh2_tensor, requires_grad=True)
    Yl_var = Variable(Yl_tensor, requires_grad=True)

    imgs = ifm((Yl_var, [Yh0_var, Yh1_var, Yh2_var]))
    # return torch.stack(imgs) / 4.
    return imgs
