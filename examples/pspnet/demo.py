import argparse

import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainercv.datasets import ade20k_semantic_segmentation_label_colors
from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import cityscapes_semantic_segmentation_label_colors
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.links import PSPNetResNet101
from chainercv.utils import read_image
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--pretrained_model')
    parser.add_argument('image')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--input_size', type=int, default=473)
    args = parser.parse_args()

    if args.dataset == 'voc2012':
        label_names = voc_semantic_segmentation_label_names
        colors = voc_semantic_segmentation_label_colors
    elif args.dataset == 'cityscapes':
        label_names = cityscapes_semantic_segmentation_label_names
        colors = cityscapes_semantic_segmentation_label_colors
    elif args.dataset == 'ade20k':
        label_names = ade20k_semantic_segmentation_label_names
        colors = ade20k_semantic_segmentation_label_colors

    if args.dataset is not None:
        n_class = len(label_names)
    else:
        label_names = None
        colors = None
        n_class = None

    input_size = (args.input_size, args.input_size)
    model = PSPNetResNet101(n_class, args.pretrained_model, input_size)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    img = read_image(args.image)
    from chainercv.transforms import resize
    print(img.shape )
    img = resize(img, (256, 512))
    labels = model.predict([img])
    label = labels[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    vis_image(img, ax=ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2, legend_handles = vis_semantic_segmentation(
        img, label, label_names, colors, ax=ax2)
    ax2.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)

    plt.show()
