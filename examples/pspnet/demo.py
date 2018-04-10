import argparse

import matplotlib.pyplot as plot
import numpy as np

import chainer
from chainercv.datasets import ade20k_semantic_segmentation_label_colors
from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import cityscapes_semantic_segmentation_label_colors
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.links import PSPNet
from chainercv.utils import read_image
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--pretrained_model', choices=['voc2012', 'cityscapes', 'ade20k'])
    # parser.add_argument('--image', '-f', type=str)
    parser.add_argument('image')
    parser.add_argument('--scales', '-s', type=float, nargs='*', default=None)
    args = parser.parse_args()

    chainer.config.train = False

    model = PSPNet(pretrained_model=args.pretrained_model)
    if args.pretrained_model == 'voc2012':
        label_names = voc_semantic_segmentation_label_names
        colors = voc_semantic_segmentation_label_colors
    elif args.pretrained_model == 'cityscapes':
        label_names = cityscapes_semantic_segmentation_label_names
        colors = cityscapes_semantic_segmentation_label_colors
    elif args.pretrained_model == 'ade20k':
        label_names = ade20k_semantic_segmentation_label_names
        colors = ade20k_semantic_segmentation_label_colors
    else:
        label_names = None
        colors = None

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    img = read_image(args.image)
    labels = model.predict([img])
    label = labels[0]

    print(type(label))
    np.save('label.npy', label)

    # fig = plot.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # vis_image(img, ax=ax1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # vis_semantic_segmentation(
    #     label, label_names, colors, ax=ax2)
    # plot.show()
