from __future__ import division

import numpy as np

import chainer
from chainer.functions import dropout
from chainer.functions import max_pooling_2d
from chainer.functions import average_pooling_2d
from chainer.functions import relu
from chainer.functions import softmax
from chainer.functions import local_response_normalization
from chainer.initializers import constant
from chainer.initializers import normal
from chainer.initializers import uniform

from chainer.links.connection.inception import Inception
from chainer.links import Linear

from chainercv.utils import download_model

from chainercv.links.model.sequential_feature_extractor import \
    SequentialFeatureExtractor
from chainercv.links.connection.convolution_2d_block import Convolution2DBlock


_imagenet_mean = np.array(
    [123.0, 117.0, 104.0], dtype=np.float32)[:, np.newaxis, np.newaxis]


class InceptionV1(SequentialFeatureExtractor):

    _models = {
    }

    def __init__(self,
                 pretrained_model=None, n_class=None, mean=None,
                 initialW=None, initial_bias=None):
        
        if initialW is None:
            # employ default initializers used in BVLC. For more detail, see
            # https://github.com/pfnet/chainer/pull/2424#discussion_r109642209
            initialW = uniform.LeCunUniform(scale=1.)
        if pretrained_model:
            initialW = constant.Zero()

        if mean is None:
            if pretrained_model in self._models:
                mean = self._models[pretrained_model]['mean']

        # TODO FIX
        self.mean = _imagenet_mean

        inception_kwargs = {'conv_init': initialW, 'bias_init': initial_bias}

        super(InceptionV1, self).__init__()
        with self.init_scope():
            self.conv1 = Convolution2DBlock(
                None, 64, 7,
                conv_kwargs={'stride': 2, 'pad': 3, 'initialW': initialW,
                             'initial_bias': initial_bias})
            self.pool1 = _max_pooling_2d
            self.pool1_lrn = _local_response_normalization
            self.conv2_reduce = Convolution2DBlock(
                None, 64, 1,
                conv_kwargs={'initialW': initialW, 'initial_bias': initial_bias})
            self.conv2 = Convolution2DBlock(
                None, 192, 3,
                conv_kwargs={'stride': 1, 'pad': 1, 'initialW': initialW,
                             'initial_bias': initial_bias})
            self.conv2_lrn = _local_response_normalization
            self.pool2 = _max_pooling_2d

            self.inc3a = Inception(None, 64, 96, 128, 16, 32, 32,
                                   **inception_kwargs)
            self.inc3b = Inception(None, 128, 128, 192, 32, 96, 64,
                                   **inception_kwargs)
            self.pool3 = _max_pooling_2d

            self.inc4a = Inception(None, 192, 96, 208, 16, 48, 64,
                                   **inception_kwargs)
            self.inc4b = Inception(None, 160, 112, 224, 24, 64, 64,
                                   **inception_kwargs)
            self.inc4c = Inception(None, 128, 128, 256, 24, 64, 64,
                                   **inception_kwargs)
            self.inc4d = Inception(None, 112, 144, 288, 32, 64, 64,
                                   **inception_kwargs)
            self.inc4e = Inception(None, 256, 160, 320, 32, 128, 128,
                                   **inception_kwargs)
            self.pool4 = _max_pooling_2d

            self.inc5a = Inception(None, 256, 160, 320, 32, 128, 128,
                                   **inception_kwargs)
            self.inc5b = Inception(None, 384, 192, 384, 48, 128, 128,
                                   **inception_kwargs)
            self.pool5 = _average_pooling_2d_k7

            self.dropout = _dropout
            self.fc = Linear(None, 1000,
                             initialW=initialW, initial_bias=initial_bias)
            self.prob = softmax
        
        if pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)



def _max_pooling_2d(x):
    return max_pooling_2d(x, ksize=3, stride=2)


def _local_response_normalization(x):
    return local_response_normalization(x, n=5, k=1, alpha=1e-4 / 5)


def _average_pooling_2d_k7(x):
    return average_pooling_2d(x, ksize=7, stride=1)


def _dropout(x):
    return dropout(x, ratio=0.4)
