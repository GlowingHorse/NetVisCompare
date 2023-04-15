# ViT channel feature visualization

import os
import numpy as np
import timm_mod
from utils.vis_ViT_neuronLayer_class import ViTLayerVisualization
import importlib

from utils.img_path_info import img_paths, corres_attr_classes


def chunks(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def topk_indices(arr, num_contrib_vis):
    contrib_neuron_indices = np.argpartition(arr, -num_contrib_vis)[-num_contrib_vis:]
    contrib_neuron_indices = np.flip(contrib_neuron_indices, 0)
    return contrib_neuron_indices


if __name__ == '__main__':
    # Param settings
    image_height = 224
    flag_save = True
    flag_color_space = False
    method_vis_neruons = ['channel-level']
    vis_mode = 'certain_img'

    iteration_steps = 101
    num_contrib_vis = 5

    optimizer_names = ['Adamax']

    # for 'channel-level'
    opt_lrs = [0.02] * 8 + [0.025] * 40

    # loss_names = [10, -10]
    loss_names = ['positive', 'negative']
    network_name = 'ViT_B_16'
    layer_name = 'blocks'

    # pretrained_model = models.googlenet(pretrained=True, transform_input=False)
    pretrained_model = timm_mod.create_model(network_name, pretrained=True)

    for i_img in range(len(img_paths)):
        img_path = img_paths[i_img]
        img_name = os.path.splitext(os.path.basename(img_path))[-2]

        attr_save_dir = '../experiments/' + network_name + \
                  '/img_info/' + img_name + \
                  '/' + 'neuronAttr'
        # layer_index = 4
        for i_optimizer_name in range(len(optimizer_names)):
            optimizer_name = optimizer_names[i_optimizer_name]

            # for layer_index in range(7, 10):
            for layer_index in [6]:
                opt_lr = opt_lrs[layer_index]

                for class_name in corres_attr_classes[i_img]:
                    attr_save_path = attr_save_dir + '/' + \
                                     class_name + '_' + layer_name + str(layer_index) + '.npy'
                    layer_attr = np.load(attr_save_path)
                    layer_attr = layer_attr[1:, :]

                    # spatial_summed_attr = np.sum(layer_attr, 0)
                    spatial_maxed_attr = np.max(layer_attr, 0)
                    # spatial_mined_attr = np.min(layer_attr, 0)
                    # pos_contrib_index_sum = topk_indices(spatial_summed_attr)
                    # neg_contrib_index_sum = topk_indices(-1 * spatial_summed_attr)
                    pos_contrib_index_max = topk_indices(spatial_maxed_attr, num_contrib_vis)
                    # neg_contrib_index_min = topk_indices(spatial_mined_attr)
                    # all_indices = list(np.concatenate((pos_contrib_index_sum, neg_contrib_index_sum,
                    #                                    pos_contrib_index_max, neg_contrib_index_min)))
                    all_indices = list(pos_contrib_index_max)
                    # batch_lst = list(chunks(range(768), 40))

                    # batch_lst = list(chunks(range(768), 20))
                    batch_lst = list(chunks(all_indices, 5))

                    for i_selected_filters in range(len(batch_lst)):
                        selected_filters = batch_lst[i_selected_filters]
                        for i_loss in range(len(loss_names)):
                            loss_name = loss_names[i_loss]

                            for method_vis_neruon in method_vis_neruons:
                                # Fully connected layer is not needed
                                layer_vis = ViTLayerVisualization(pretrained_model,
                                                                  selected_filters=selected_filters,
                                                                  image_height=image_height,
                                                                  layer_name=layer_name, layer_index=layer_index,
                                                                  gen_dir='../experiments',
                                                                  optimizer_name=optimizer_name, opt_lr=opt_lr,
                                                                  loss_name=loss_name,
                                                                  network_name=network_name,
                                                                  method_vis_neruon=method_vis_neruon,
                                                                  vis_mode=vis_mode)

                                # Layer visualization with pytorch hooks
                                layer_vis.visualize_layer_with_hooks(flag_save=flag_save,
                                                                     flag_color_space=flag_color_space,
                                                                     iteration_steps=iteration_steps)

                                print('Img name {}, class name {}'.format(img_name, class_name))
                                print('Opt name: {}, layer_name: {}, layer index: {}, '
                                      'i_selected_filters: {}, loss_name: {}, method_vis_neruon {}'.
                                      format(optimizer_name, layer_name, layer_index,
                                             str(i_selected_filters), loss_name, method_vis_neruon))
                                print('')

