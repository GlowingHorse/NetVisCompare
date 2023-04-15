import matplotlib.pyplot as plt

# Using Agg is much faster than nothing or TkAgg
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import numpy as np
import os


def create_no_exist_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot(heatmap, save_directory=None, save_img_name=None,
         xi=None, cmap='RdBu_r', cmap2='seismic', alpha=0.3,
         flag_save=False):
    # heatmap'size is same as input original image
    plt.ioff()
    # plt.ion()
    fig = plt.figure(1, figsize=[2.24, 2.24], dpi=100, frameon=False)

    axis = plt.Axes(fig, [0., 0., 1., 1.])
    axis.set_axis_off()
    fig.add_axes(axis)

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, heatmap.shape[1]+dx, dx)
    yy = np.arange(0.0, heatmap.shape[0]+dy, dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap(cmap2).copy()
    cmap_xi.set_bad(alpha=0)
    overlay = xi
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, 2)
    # axis.imshow(heatmap, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    axis.imshow(heatmap, extent=extent, interpolation='none', cmap=cmap)
    axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    # factorization_method = findall('[A-Z]', factorization_method)
    # factorization_method = ''.join(factorization_method)
    if flag_save:
        plt.savefig(save_directory + '/' + save_img_name)  # 'RdBu_r' 'hot'
    else:
        plt.show()
    plt.close(1)
    # print()


def plot_grid_hm(intensity_map, save_directory, save_img_name):
    plt.ioff()
    fig = plt.figure(1, figsize=[2.24, 2.24], dpi=100, frameon=False)

    axis = plt.Axes(fig, [0., 0., 1., 1.])
    axis.set_axis_off()
    fig.add_axes(axis)

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, intensity_map.shape[1] + dx, dx)
    yy = np.arange(0.0, intensity_map.shape[0] + dy, dy)

    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap = 'RdBu_r'

    axis.imshow(intensity_map, extent=extent, interpolation='none', cmap=cmap)
    plt.savefig(save_directory + '/' + save_img_name)
    # plt.show()
    # print()
