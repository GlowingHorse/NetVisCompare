
# CNN channel feature visualization

import os
import numpy as np
from torchvision import models
from utils.vis_CNN_neuronLayer_class import CNNLayerVisualization
from utils.img_path_info import img_paths, corres_attr_classes


def chunks(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def topk_indices(arr):
    num_contrib_channel = 10
    contrib_neuron_indices = np.argpartition(arr, -num_contrib_channel)[-num_contrib_channel:]
    contrib_neuron_indices = np.flip(contrib_neuron_indices, 0)
    return contrib_neuron_indices


if __name__ == '__main__':
    # Param settings
    image_height = 224
    flag_debug = True
    flag_save = True

    method_vis_neruon = 'channel-level'

    iteration_steps = 51

    optimizer_names = ['Adamax']
    network_name = 'resnet50'

    # layer_names = {'conv1': [0],
    #                'layer1': [0, 1, 2],
    #                'layer2': list(range(4)),
    #                'layer3': [0, 1, 2, 3, 4, 5],
    #                'layer4': [0, 1, 2]}
    layer_names = {'layer4': [0]}

    opt_lrs = [0.011]*59

    for i_img in range(len(img_paths)):
        img_path = img_paths[i_img]
        img_name = os.path.splitext(os.path.basename(img_path))[-2]
        attr_save_dir = '../experiments/' + network_name + \
                        '/img_info/' + img_name + \
                        '/' + 'neuronAttr'
        for i_optimizer_name in range(len(optimizer_names)):
            optimizer_name = optimizer_names[i_optimizer_name]
            opt_lr = opt_lrs[i_optimizer_name]

            for layer_name in layer_names.keys():
                layer_indices = layer_names[layer_name]

                for i_layer_index in range(len(layer_indices)):
                    for class_name in corres_attr_classes[i_img]:
                        layer_index = layer_indices[i_layer_index]
                        # channel_num = channel_nums[layer_name][layer_index]

                        attr_save_path = attr_save_dir + '/' + class_name + '_' + layer_name + \
                                         '_' + str(layer_index) + '.npy'
                        layer_attr = np.load(attr_save_path)
                        # spatial_summed_attr = np.sum(layer_attr, (1, 2))
                        spatial_maxed_attr = np.max(layer_attr, (1, 2))
                        # spatial_mined_attr = np.min(layer_attr, (1, 2))
                        # pos_contrib_index_sum = topk_indices(spatial_summed_attr)
                        # neg_contrib_index_sum = topk_indices(-1 * spatial_summed_attr)
                        pos_contrib_index_max = topk_indices(spatial_maxed_attr)
                        # neg_contrib_index_min = topk_indices(spatial_mined_attr)
                        # all_indices = list(np.concatenate((pos_contrib_index_sum, neg_contrib_index_sum,
                        #                                    pos_contrib_index_max, neg_contrib_index_min)))
                        # all_indices = list(set(all_indices))

                        all_indices = list(pos_contrib_index_max)

                        batch_lst = list(chunks(all_indices, 10))

                        for i_selected_filters in range(len(batch_lst)):
                            model = getattr(models, network_name)(pretrained=True)
                            if i_selected_filters < -1:
                                pass
                            else:
                                selected_filters = list(batch_lst[i_selected_filters])
                                layer_vis = CNNLayerVisualization(model, selected_filters=selected_filters,
                                                                  image_height=image_height,
                                                                  layer_name=layer_name, layer_index=layer_index,
                                                                  gen_dir='../experiments',
                                                                  optimizer_name=optimizer_name, opt_lr=opt_lr,
                                                                  network_name=network_name,
                                                                  method_vis_neruon=method_vis_neruon)

                                # Layer visualization with pytorch hooks
                                layer_vis.visualize_layer_with_hooks(flag_debug=flag_debug, flag_save=flag_save,
                                                                     iteration_steps=iteration_steps)

                                print('Opt name: {}, layer_name: {}, layer index: {}, '
                                      'i_selected_filters and num: {} {}'.
                                      format(optimizer_name, layer_name, layer_index,
                                             str(i_selected_filters), str(len(selected_filters))))

                                del layer_vis, selected_filters
                                print('')

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
