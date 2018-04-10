import numpy as np
import unittest

from chainer import testing
from chainercv.links.model.pspnet.transforms import convolution_crop


class TestConvolutionCrop(unittest.TestCase):

    def test_convolution_crop(self):
        img = np.random.uniform(size=(3, 48, 64))

        size = (12, 15)
        stride = (8, 4)
        crop_imgs, param = convolution_crop(img, size, stride, return_param=True)

        canvas = np.zeros(img.shape)
        count = np.zeros(img.shape[1:])
        for i in range(len(crop_imgs)):
            assert crop_imgs[i].shape == (3,) + size
            crop_y_slice = param['crop_y_slices'][i]
            crop_x_slice = param['crop_x_slices'][i]
            y_slice = param['y_slices'][i]
            x_slice = param['x_slices'][i]
            canvas[:, y_slice, x_slice] += crop_imgs[i][:, crop_y_slice, crop_x_slice]
            count[y_slice, x_slice] += 1
        canvas = canvas / count[None]

        np.testing.assert_almost_equal(canvas, img)


testing.run_module(__name__, __file__)
