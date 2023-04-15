# NetVisCompare
Implementation for "Visualization Comparison of Vision Transformers and Convolutional Neural Networks".

The optimization-based visualization method can be used to understand the semantic meanings of network representations.

AttrVis Quickstart
===
## Installation
0. It is safer to create a virtual environment before installing libraries.
1. Install necessary libraries listed in *requirements.txt*.

## Usage
1. Run */s_cal_attr/\*.py* to compute attribution scores.
2. Run */s_vis_cnn/\*.py* to generate CNN visualizations.
3. Run */s_vis_vit/\*.py* to generate ViT visualizations.

## For params tests
- Check */utils/transform_robust.py* to select indirect regularization techniques.
- Check */utils_params/random_params.py* for information about frequency domain optimization

## Others
- We are very grateful to Robert Geirhos for providing us with the texture-bias dataset. If you need to use this dataset, please refer to: [**texture-vs-shape**](https://github.com/rgeirhos/texture-vs-shape).
- Next, we plan to further refactor our codes to improve usability and readability, and upload the reconstructed codes to this repository.
