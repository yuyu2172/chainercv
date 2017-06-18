import numpy as np

import chainer
from chainer.initializers import constant
from chainer.initializers import normal

from chainercv.links.model.resnet.resnet import ResNet
from chainercv.utils import download_model

# RGB order
_imagenet_mean = np.array(
    [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]


class ResNet50(ResNet):

    _models = {}

    def __init__(self, pretrained_model=None, n_class=None,
                 features='prob', initialW=None,
                 mean=_imagenet_mean, do_ten_crop=False):
        if n_class is None:
            if (pretrained_model is None and
                    all([feature not in ['fc6', 'prob']
                         for feature in features])):
                # fc8 layer is not used in this case.
                pass
            elif pretrained_model not in self._models:
                raise ValueError(
                    'The n_class needs to be supplied as an argument.')
            else:
                n_class = self._models[pretrained_model]['n_class']

        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            if initialW is None:
                initialW = constant.Zero()
        else:
            # Employ default initializers used in the original paper.
            if initialW is None:
                initialW = normal.HeNormal(scale=1.)

        super(ResNet50, self).__init__(
            [3, 4, 6, 3], n_class, features, initialW, mean, do_ten_crop)

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)


if __name__ == '__main__':
    model = ResNet50()
