import os
import numpy as np
from struct import pack, unpack
import scipy.io as sio
import struct
import glob
import pathlib
import re


def get_proj_abs_dir():
    # e.g., '\\PycharmProjects\\SwingGar'
    abs_dir = str(pathlib.Path().resolve())
    if abs_dir.endswith('TorchVisNet'):
        return abs_dir
    else:
        indices = re.finditer('TorchVisNet', abs_dir)
        for i in indices:
            span_range = i.span()
            abs_dir = abs_dir[0:span_range[1]]
            break
        return abs_dir


def create_dir_if_no_exist(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
