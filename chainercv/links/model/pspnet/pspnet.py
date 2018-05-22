from __future__ import division

from math import ceil
import numpy as np
import six
import warnings

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv import transforms
from chainercv import utils
from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet import ResBlock
from chainercv.links.model.pspnet.transforms import convolution_crop


# RGB order
_imagenet_mean = np.array(
    [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]


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
        kh = (feat_size[0] // np.array(pyramids))
        kw = (feat_size[1] // np.array(pyramids))
        self.ksizes = list(zip(kh, kw))

    def __call__(self, x):
        ys = [x]
        h, w = x.shape[2:]
        for f, ksize in zip(self, self.ksizes):
            y = F.average_pooling_2d(x, ksize, ksize)  # Pad should be 0!
            y = f(y)  # Reduce num of channels
            y = F.resize_images(y, (h, w))
            ys.append(y)
        return F.concat(ys, axis=1)


class DilatedFCN(chainer.Chain):

    def __init__(self, n_block, initialW, mid_downsample, bn_kwargs=None):
        stride_first = not mid_downsample
        super(DilatedFCN, self).__init__()
        with self.init_scope():
            self.conv1_1 = Conv2DBNActiv(
                None, 64, 3, 2, 1, 1,
                initialW=initialW, bn_kwargs=bn_kwargs)
            self.conv1_2 = Conv2DBNActiv(
                64, 64, 3, 1, 1, 1, initialW=initialW, bn_kwargs=bn_kwargs)
            self.conv1_3 = Conv2DBNActiv(
                64, 128, 3, 1, 1, 1, initialW=initialW, bn_kwargs=bn_kwargs)
            self.res2 = ResBlock(
                n_block[0], 128, 64, 256, 1, 1,
                initialW, bn_kwargs, stride_first)
            self.res3 = ResBlock(
                n_block[1], 256, 128, 512, 2, 1,
                initialW, bn_kwargs, stride_first)
            self.res4 = ResBlock(
                n_block[2], 512, 256, 1024, 1, 2,
                initialW, bn_kwargs, stride_first)
            self.res5 = ResBlock(
                n_block[3], 1024, 512, 2048, 1, 4,
                initialW, bn_kwargs, stride_first)

    def __call__(self, x):
        h = self.conv1_3(self.conv1_2(self.conv1_1(x)))
        h = F.max_pooling_2d(h, 3, 2, 1)
        h = self.res2(h)
        h = self.res3(h)
        h1 = self.res4(h)
        h2 = self.res5(h1)
        return h1, h2


class PSPNet(chainer.Chain):

    """Pyramid Scene Parsing Network

    This Chain supports any depth of ResNet and any pyramid levels for
    the pyramid pooling module (PPM).

    When you specify the path of a pre-trained chainer model serialized as
    a :obj:`.npz` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    When a string in prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows:

    * :obj:`voc2012`: Loads weights trained with the trainval split of \
        PASCAL VOC2012 Semantic Segmentation Dataset.
    * :obj:`cityscapes`: Loads weights trained with Cityscapes dataset.
    * :obj:`ade20k`: Loads weights trained with ADE20K dataset.

    ..note::

        Somehow

    Args:
        n_class (int): The number of channels in the last convolution layer.
        input_size (int or iterable of ints): The input image size. If a
            single integer is given, it's treated in the same way as if
            a tuple of (input_size, input_size) is given. If an iterable object
            is given, it should mean (height, width) of the input images.
        n_blocks (list of int): Numbers of layers in ResNet. Typically,
            [3, 4, 23, 3] for ResNet101 (used for PASCAL VOC2012 and
            Cityscapes in the original paper) and [3, 4, 6, 3] for ResNet50
            (used for ADE20K datset in the original paper).
        pyramids (list of int): The number of division to the feature map in
            each pyramid level. The length of this list will be the number of
            levels of pyramid in the pyramid pooling module. In each pyramid,
            an average pooling is applied to the feature map with the kernel
            size of the corresponding value in this list.
        mid_stride (bool): If True, spatial dimention reduction in bottleneck
            modules in ResNet part will be done at the middle 3x3 convolution.
            It means that the stride of the middle 3x3 convolution will be two.
            Otherwise (if it's set to False), the stride of the first 1x1
            convolution in the bottleneck module will be two as in the original
            ResNet and Deeplab v2.
        mean (numpy.ndarray): A value to be subtracted from an image
            in :meth:`prepare`.
        comm (chainermn.communicator or None): If a ChainerMN communicator is
            given, it will be used for distributed batch normalization during
            training. If None, all batch normalization links will not share
            the input vectors among GPUs before calculating mean and variance.
            The original PSPNet implementation uses distributed batch
            normalization.
        pretrained_model (str): The destination of the pre-trained
            chainer model serialized as a :obj:`.npz` file.
            If this is one of the strings described
            above, it automatically loads weights stored under a directory
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/models/`,
            where :obj:`$CHAINER_DATASET_ROOT` is set as
            :obj:`$HOME/.chainer/dataset` unless you specify another value
            by modifying the environment variable.

    """

    def __init__(self, n_block, n_class, input_size,
                 initialW=None, comm=None, compute_aux=False):
        super(PSPNet, self).__init__()
        if comm is not None:
            bn_kwargs = {'comm': comm}
        else:
            bn_kwargs = {}
        if initialW is None:
            initialW = chainer.initializers.HeNormal()

        mid_stride = True
        pyramids = [6, 3, 2, 1]
        mean = np.array([123.68, 116.779, 103.939])

        if not isinstance(input_size, (list, tuple)):
            input_size = (int(input_size), int(input_size))

        # scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
        self.scales = [1]
        self.mean = mean.astype(np.float32)
        self.compute_aux = compute_aux
        self.input_size = input_size
        with self.init_scope():
            self.trunk = DilatedFCN(n_block=n_block, initialW=initialW,
                                    mid_downsample=mid_stride, bn_kwargs=bn_kwargs)

            # To calculate auxirally loss
            # Perhaps, this should not be placed here
            self.aux_conv1 = Conv2DBNActiv(None, 512, 3, 1, 1, initialW=initialW)
            self.aux_conv2 = L.Convolution2D(
                512, n_class, 3, 1, 1, False, initialW)

            # Main branch
            feat_size = (input_size[0] // 8, input_size[1] // 8)
            self.ppm = PyramidPoolingModule(2048, feat_size, pyramids,
                                            initialW=initialW, bn_kwargs=bn_kwargs)
            self.main_conv1 = Conv2DBNActiv(4096, 512, 3, 1, 1,
                                            initialW=initialW)
            self.main_conv2 = L.Convolution2D(
                512, n_class, 1, 1, 0, False, initialW)

    @property
    def n_class(self):
        return self.main_conv2.out_channels

    def __call__(self, x):
        """Forward computation of PSPNet

        Args:
            x: Input array or Variable.

        Returns:
            Training time: it returns the outputs from auxiliary branch and the
                main branch. So the returned value is a tuple of two Variables.
            Inference time: it returns the output of the main branch. So the
                returned value is a sinle Variable which forms
                ``(N, n_class, H, W)`` where ``N`` is the batchsize and
                ``n_class`` is the number of classes specified in the
                constructor. ``H, W`` is the input image size.

        """
        if self.compute_aux:
            aux, h = self.trunk(x)
            aux = F.dropout(self.aux_conv1(aux), ratio=0.1)
            aux = self.aux_conv2(aux)
            aux = F.resize_images(aux, x.shape[2:])
        else:
            h = self.trunk(x)

        h = self.ppm(h)
        h = F.dropout(self.main_conv1(h), ratio=0.1)
        h = self.main_conv2(h)
        h = F.resize_images(h, x.shape[2:])

        if chainer.config.train:
            return aux, h
        else:
            return h

    def _tile_predict(self, img):
        # Prepare should be made
        if self.mean is not None:
            img = img - self.mean[:, None, None]
            # if self._use_pretrained_model:
            if True:
                pass
                # pretrained model is trained using BGR images
                # img = img[::-1]
        ori_rows, ori_cols = img.shape[1:]
        long_size = max(ori_rows, ori_cols)

        # When padding input patches is needed
        if long_size > max(self.input_size):
            stride_rate = 2 / 3.
            stride = (int(ceil(self.input_size[0] * stride_rate)),
                      int(ceil(self.input_size[1] * stride_rate)))

            imgs, param = convolution_crop(img, self.input_size, stride, return_param=True)

            count = self.xp.zeros((1, ori_rows, ori_cols), dtype=np.float32)
            pred = self.xp.zeros((1, self.n_class, ori_rows, ori_cols), dtype=np.float32)
            N = len(param['y_slices'])
            for i in range(N):
                img_i = imgs[i:i+1]
                y_slice = param['y_slices'][i]
                x_slice = param['x_slices'][i]
                crop_y_slice = param['crop_y_slices'][i]
                crop_x_slice = param['crop_x_slices'][i]

                var = chainer.Variable(self.xp.asarray(img_i))
                with chainer.using_config('train', False):
                    scores_i = F.softmax(self.__call__(var)).data
                assert scores_i.shape[2:] == var.shape[2:]
                pred[0, :, y_slice, x_slice] += scores_i[0, :, crop_y_slice, crop_x_slice]

                # Horizontal flip
                flipped_var = chainer.Variable(self.xp.asarray(img_i[:, :, :, ::-1]))
                with chainer.using_config('train', False):
                    flipped_scores_i = F.softmax(self.__call__(flipped_var)).data
                # Flip horizontally flipped score maps again
                flipped_scores_i = flipped_scores_i[:, :, :, ::-1]
                pred[0, :, y_slice, x_slice] +=\
                    flipped_scores_i[0, :, crop_y_slice, crop_x_slice]
                count[0, y_slice, x_slice] += 2

            # scores[N:] = scores[N:][:, :, :, ::-1]

            # for i in range(N):
            #     y_slice = param['y_slices'][i]
            #     x_slice = param['x_slices'][i]
            #     crop_y_slice = param['crop_y_slices'][i]
            #     crop_x_slice = param['crop_x_slices'][i]

            #     score_0 = scores[i, :, crop_y_slice, crop_x_slice]
            #     pred[0, :, y_slice, x_slice] += score_0
            #     score_1 = scores[N + i, :, crop_y_slice, crop_x_slice]
            #     pred[0, :, y_slice, x_slice] += score_1
            #     count[0, y_slice, x_slice] += 2
            # print('aaaa')
            # hh = int(ceil((ori_rows - self.input_size[0]) / stride[0])) + 1
            # ww = int(ceil((ori_cols - self.input_size[1]) / stride[1])) + 1
            # for yy in six.moves.xrange(hh):
            #     for xx in six.moves.xrange(ww):
            #         sy, sx = yy * stride[0], xx * stride[1]
            #         ey, ex = sy + self.input_size[0], sx + self.input_size[1]
            #         img_sub = img[:, sy:ey, sx:ex]
            #         img_sub, pad_h, pad_w = self._pad_img(img_sub)

            #         # Take average of pred and pred from flipped image
            #         psub1 = self._predict(img_sub[np.newaxis])
            #         psub2 = self._predict(img_sub[np.newaxis, :, :, ::-1])
            #         psub = (psub1 + psub2[:, :, :, ::-1]) / 2.

            #         if sy + self.input_size[0] > ori_rows:
            #             psub = psub[:, :, :-pad_h, :]
            #         if sx + self.input_size[1] > ori_cols:
            #             psub = psub[:, :, :, :-pad_w]
            #         pred[:, :, sy:ey, sx:ex] = psub
            #         count[sy:ey, sx:ex] += 1
            score = pred / count[:, None]
        else:
            raise ValueError("Not supported\n")
            # img, pad_h, pad_w = self._pad_img(img)
            # pred1 = self._predict(img[np.newaxis])
            # pred2 = self._predict(img[np.newaxis, :, :, ::-1])
            # pred = (pred1 + pred2[:, :, :, ::-1]) / 2.
            # score = pred[
            #     :, :, :self.input_size[0] - pad_h, :self.input_size[1] - pad_w]
        score = F.resize_images(score, (ori_rows, ori_cols))[0].data
        return score

    def predict(self, imgs):
        """Conduct semantic segmentation from images.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images should be in CHW order.

        Returns:
            list of numpy.ndarray: List of predictions from each image in the
                input list. Note that if you specified ``argmax=True``, each
                prediction is resulting integer label and the number of
                dimensions is two (:math:`(H, W)`). Otherwise, the output will
                be a probability map calculated by the model and its number of
                dimensions will be three (:math:`(C, H, W)`).

        """
        labels = []
        for img in imgs:
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                if self.scales is not None:
                    scores = _multiscale_predict(self._tile_predict, img, self.scales)
                else:
                    scores = self._tile_predict(img)
            labels.append(chainer.cuda.to_cpu(
                self.xp.argmax(scores, axis=0).astype(np.int32)))
        return labels


class PSPNetResNet101(PSPNet):

    _models = {
        'cityscapes': {
            'n_class': 19,
            'input_size': (713, 713),
            'feat_size': 90,
            'url': 'https://github.com/mitmul/chainer-pspnet/releases/download'
                   '/ChainerCV_PSPNet/pspnet101_cityscapes_713_reference.npz'
        },
        # 'voc2012': {
        #     'n_class': 21,
        #     'input_size': (473, 473),
        #     'feat_size': 60,
        #     'url': 'https://github.com/mitmul/chainer-pspnet/releases/download'
        #     '/ChainerCV_PSPNet/pspnet101_VOC2012_473_reference.npz'
        # }
    }

    def __init__(self, n_class=None, pretrained_model=None,
                 input_size=None, mean=None,
                 initialW=None, comm=None, compute_aux=True):
        param, path = utils.prepare_pretrained_model(
            {'n_class': n_class, 'input_size': input_size},
            pretrained_model, self._models,
            {'input_size': (713, 713)})

        n_block = [3, 4, 23, 3]
        super(PSPNetResNet101, self).__init__(
            n_block, param['n_class'], param['input_size'], initialW, comm, compute_aux)

        if path:
            chainer.serializers.load_npz(path, self)


def _multiscale_predict(predict_method, img, scales):
    orig_H, orig_W = img.shape[1:]
    scores = []
    orig_img = img
    for scale in scales:
        img = orig_img.copy()
        if scale != 1.0:
            img = transforms.resize(
                img, (int(orig_H * scale), int(orig_W * scale)))
        # This method should return scores
        y = predict_method(img)[None]
        assert y.shape[2:] == img.shape[1:]

        if scale != 1.0:
            y = F.resize_images(y, (orig_H, orig_W)).data
        scores.append(y)
    xp = chainer.cuda.get_array_module(scores[0])
    scores = xp.stack(scores)
    return scores.mean(0)[0]  # (C, H, W)
