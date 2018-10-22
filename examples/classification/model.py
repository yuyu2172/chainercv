from __future__ import division

from math import ceil
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.experimental.links.model.pspnet.transforms import \
    convolution_crop
from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet import ResBlock
from chainercv.links import PickableSequentialChain
from chainercv import transforms
from chainercv import utils

from chainercv.links.model.resnet.resnet import _imagenet_mean


class PyramidPoolingModule(chainer.ChainList):

    def __init__(self, in_channels, feat_size, pyramids,
                 initialW=None,
                 bn_kwargs=None):
        out_channels = in_channels // len(pyramids)
        super(PyramidPoolingModule, self).__init__(
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW,
                bn_kwargs=bn_kwargs),
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW,
                bn_kwargs=bn_kwargs),
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW,
                bn_kwargs=bn_kwargs),
            Conv2DBNActiv(
                in_channels, out_channels, 1, 1, 0, 1, initialW=initialW,
                bn_kwargs=bn_kwargs)
        )
        kh = feat_size[0] // np.array(pyramids)
        kw = feat_size[1] // np.array(pyramids)
        self.ksizes = list(zip(kh, kw))

    def __call__(self, x):
        ys = [x]
        H, W = x.shape[2:]
        for f, ksize in zip(self, self.ksizes):
            y = F.average_pooling_2d(x, ksize, ksize)
            y = f(y)
            y = F.resize_images(y, (H, W))
            ys.append(y)
        return F.concat(ys, axis=1)


class PSPNetBackbone(PickableSequentialChain):

    _blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
    }

    def __init__(self, n_class, n_layer, initialW=None, bn_kwargs={}):
        self.mean = _imagenet_mean

        n_block = self._blocks[n_layer]

        if initialW is None:
            initialW = chainer.initializers.HeNormal(scale=1., fan_option='fan_out')

        fc_kwargs = {}
        fc_kwargs['initialW'] = chainer.initializers.Normal(scale=0.01)
        super(PSPNetBackbone, self).__init__()
        with self.init_scope():
            self.conv1_1 = Conv2DBNActiv(
                None, 64, 3, 2, 1, 1,
                initialW=initialW, bn_kwargs=bn_kwargs)
            self.conv1_2 = Conv2DBNActiv(
                64, 64, 3, 1, 1, 1, initialW=initialW, bn_kwargs=bn_kwargs)
            self.conv1_3 = Conv2DBNActiv(
                64, 128, 3, 1, 1, 1, initialW=initialW, bn_kwargs=bn_kwargs)
            self.pool1 = lambda x: F.max_pooling_2d(
                x, ksize=3, stride=2, pad=1)
            self.res2 = ResBlock(
                n_block[0], 128, 64, 256, 1, 1,
                initialW, bn_kwargs, stride_first=False)
            self.res3 = ResBlock(
                n_block[1], 256, 128, 512, 2, 1,
                initialW, bn_kwargs, stride_first=False)
            self.res4 = ResBlock(
                n_block[2], 512, 256, 1024, 2, 1,
                initialW, bn_kwargs, stride_first=False)
            self.res5 = ResBlock(
                n_block[3], 1024, 512, 2048, 2, 1,
                initialW, bn_kwargs, stride_first=False)
            self.pool5 = _global_average_pooling_2d
            self.fc6 = L.Linear(None, n_class, **fc_kwargs)
            self.prob = F.softmax


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = h.reshape((n, channel))
    return h
