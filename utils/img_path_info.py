
from utils.IO import get_proj_abs_dir

abs_dir = get_proj_abs_dir()
img_paths = []
corres_attr_classes = []

img_paths = img_paths + \
    [abs_dir + '/data/images/cat-flower.jpg']
corres_attr_classes = corres_attr_classes + \
    [['tiger cat']]




