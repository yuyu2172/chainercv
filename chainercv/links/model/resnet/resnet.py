from __future__ import division

import collections

import chainer.functions as F
import chainer.links as L

from chainercv.transforms import ten_crop

from chainercv.links.model.resnet.building_block import BuildingBlock
from chainercv.links.model.sequential_feature_extraction_chain import \
    SequentialFeatureExtractionChain


class ResNet(SequentialFeatureExtractionChain):

    def __init__(self, blocks, n_class=None,
                 feature_names='prob', initialW=None,
                 mean=None, do_ten_crop=False):

        def _getattr(name):
            return getattr(self, name, None)

        kwargs = {'initialW': initialW}
        link_generators = {
            'conv1': lambda: L.Convolution2D(3, 64, 7, 2, 3, **kwargs),
            'bn1': lambda: L.BatchNormalization(64),
            'res2': lambda: BuildingBlock(blocks[0], 64, 64, 256, 1, **kwargs),
            'res3': lambda: BuildingBlock(
                blocks[1], 256, 128, 512, 2, **kwargs),
            'res4': lambda: BuildingBlock(
                blocks[2], 512, 256, 1024, 2, **kwargs),
            'res5': lambda: BuildingBlock(
                blocks[3], 1024, 512, 2048, 2, **kwargs),
            'fc6': lambda: L.Linear(2048, 1000)
        }

        super(ResNet, self).__init__(
            feature_names,
            link_generators,
            mean, ten_crop)

    @property
    def default_functions(self):
        def _getattr(name):
            return getattr(self, name, None)

        return collections.OrderedDict([
            ('conv1', [_getattr('conv1'), _getattr('bn1'), F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, ksize=3, stride=2)]),
            ('res2', [_getattr('res2')]),
            ('res3', [_getattr('res3')]),
            ('res4', [_getattr('res4')]),
            ('res5', [_getattr('res5')]),
            ('pool5', [_global_average_pooling_2d]),
            ('fc6', [_getattr('fc6')]),
            ('prob', [F.softmax]),
        ])


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = h.reshape(n, channel)
    return h
