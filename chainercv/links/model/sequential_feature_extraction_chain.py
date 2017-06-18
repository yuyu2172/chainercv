import copy

import chainer
from chainer import cuda

from chainercv.transforms import center_crop
from chainercv.transforms import ten_crop
from chainercv.transforms import scale


class SequentialFeatureExtractionChain(chainer.Chain):

    def __init__(self, feature_names, default_functions, link_generators,
                 mean, do_ten_crop=False):
        self._default_functions = default_functions
        self.mean = mean
        self.do_ten_crop = do_ten_crop

        if (not isinstance(feature_names, str) and
                all([isinstance(feature, str) for feature in feature_names])):
            return_tuple = True
        else:
            return_tuple = False
            feature_names = [feature_names]
        self._return_tuple = return_tuple
        self._feature_names = feature_names

        super(SequentialFeatureExtractionChain, self).__init__()

        with self.init_scope():
            for name, link_gen in link_generators.items():
                if name not in self.unused_functions:
                    setattr(self, name, link_gen())

    @property
    def functions(self):
        _functions = copy.copy(self._default_functions)
        for unused_function in self.unused_functions:
            _functions.pop(unused_function)
        return _functions

    @property
    def unused_functions(self):
        if any([name not in self._default_functions for
                name in self._feature_names]):
            raise ValueError('Elements of `features` shuold be one of '
                             '{}.'.format(funcs.keys()))

        # Remove all functions that are not necessary.
        pop_funcs = False
        features = list(self._feature_names)
        _unused_functions = []
        for name in list(self._default_functions.keys()):
            if pop_funcs:
                _unused_functions.append(name)

            if name in features:
                features.remove(name)
            if len(features) == 0:
                pop_funcs = True
        return _unused_functions

    def __call__(self, x):
        """Forward VGG16.

        Args:
            x (~chainer.Variable): Batch of image variables.

        Returns:
            Variable or tuple of Variable:
            A batch of features or tuple of them.
            The features to output are selected by :obj:`features` option
            of :meth:`__init__`.

        """
        activations = {}
        h = x
        for name, funcs in self.functions.items():
            for func in funcs:
                h = func(h)
            if name in self._feature_names:
                activations[name] = h

        if self._return_tuple:
            activations = tuple(
                [activations[name] for name in activations.keys()])
        else:
            activations = list(activations.values())[0]
        return activations

    def _prepare(self, img):
        """Transform an image to the input for VGG network.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        img = scale(img, size=256)
        img = img - self.mean

        return img

    def _average_ten_crop(self, y):
        if y.ndim != 2:
            raise ValueError(
                'Ten crop can be used only for features with two dimensions.')

        xp = chainer.cuda.get_array_module(y)
        n = y.shape[0] // 10
        y_shape = y.shape[1:]
        y = y.reshape((n, 10) + y_shape)
        y = xp.sum(y, axis=1) / 10
        return y

    def predict(self, imgs):
        """Predict features from images.

        When :obj:`self.do_ten_crop == True`, this extracts features from
        patches that are ten-cropped from images.
        Otherwise, this extracts features from center-crop of the images.

        When using patches from ten-crop, the features for each crop
        is averaged to compute one feature.
        Ten-crop mode is only supported for calculation of features
        :math:`fc6, fc7, fc8, prob`.

        Given :math:`N` input images, this outputs a batched array with
        batchsize :math:`N`.

        Args:
            imgs (iterable of numpy.ndarray): Array-images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
            Variable or tuple of Variable:
            A batch of features or tuple of them.
            The features to output are selected by :obj:`features` option
            of :meth:`__init__`.

        """
        imgs = [self._prepare(img) for img in imgs]
        if self.do_ten_crop:
            imgs = [ten_crop(img, (224, 224)) for img in imgs]
        else:
            imgs = [center_crop(img, (224, 224)) for img in imgs]
        imgs = self.xp.asarray(imgs).reshape(-1, 3, 224, 224)

        with chainer.function.no_backprop_mode():
            imgs = chainer.Variable(imgs)
            activations = self(imgs)

        if isinstance(activations, tuple):
            output = []
            for activation in activations:
                activation = activation.data
                if self.do_ten_crop:
                    activation = self._average_ten_crop(activation)
                output.append(cuda.to_cpu(activation))
            output = tuple(output)
        else:
            output = cuda.to_cpu(activations.data)
            if self.do_ten_crop:
                output = self._average_ten_crop(output)

        return output
