import unittest

import numpy as np

from chainercv.evaluations import eval_detection_voc

from chainer import testing


@testing.parameterize(
    {'minoverlap': 0.5,
     'rec': np.array([0., 0., 1.]),
     'prec': np.array([0., 0., 1. / 3.])},
    {'minoverlap': 0.97,
     'rec': np.array([0., 0., 0.]),
     'prec': np.array([0., 0., 0.])}
)
class TestEvalDetectionVOCOneBbox(unittest.TestCase):

    def test_eval_detection_voc_one_bbox(self):
        bboxes = [np.array([
            [0., 0., 1., 1.], [0., 0., 2., 2.], [0.3, 0.3, 0.5, 0.5]])]
        labels = [np.array([0, 0, 0])]
        confs = [np.array([0.8, 0.9, 1.])]
        gt_bboxes = [np.array([[0., 0., 1., 0.9]])]
        gt_labels = [np.array([0])]
        # iou is [0.95, 0.422, 0.3789]

        results = eval_detection_voc(
            bboxes, labels, confs, gt_bboxes, gt_labels,
            n_class=1, minoverlap=self.minoverlap)
        np.testing.assert_equal(results[0]['recall'], self.rec)
        np.testing.assert_equal(results[0]['precision'], self.prec)


@testing.parameterize(
    {'use_07_metric': False,
     'ap0': 0.25,
     'ap1': 0.5},
    {'use_07_metric': True,
     'ap0': 0.5 / 11. * 6,
     'ap1': 0.5},
)
class TestEvalDetectionVOCMultipleBboxes(unittest.TestCase):

    minoverlap = 0.4
    rec0 = np.array([0.0, 0.5, 0.5])
    prec0 = np.array([0., 0.5, 1. / 3.])
    rec1 = np.array([0., 1.])
    prec1 = np.array([0., 0.5])

    def test_eval_detection_voc(self):
        bboxes = [
            np.array([[0., 4., 1., 5.], [0., 0., 1., 1.]]),
            np.array([[0., 0., 2., 2.], [2., 2., 3., 3.], [5., 5., 7., 7.]])
        ]
        labels = [np.array([0, 0]), np.array([0, 1, 1])]
        confs = [np.array([1., 0.9]), np.array([0.7, 0.6, 0.8])]
        gt_bboxes = [
            np.array([[0., 0., 1., 1.], [1., 0., 4., 4.]]),
            np.array([[2., 2., 3., 3.]])
        ]
        gt_labels = [np.array([0, 0]), np.array([1])]

        results = eval_detection_voc(
            bboxes, labels, confs, gt_bboxes, gt_labels,
            n_class=3, minoverlap=self.minoverlap,
            use_07_metric=self.use_07_metric)
        np.testing.assert_equal(results[0]['recall'], self.rec0)
        np.testing.assert_equal(results[0]['precision'], self.prec0)
        np.testing.assert_almost_equal(results[0]['ap'], self.ap0)
        np.testing.assert_equal(results[1]['recall'], self.rec1)
        np.testing.assert_equal(results[1]['precision'], self.prec1)
        np.testing.assert_almost_equal(results[1]['ap'], self.ap1)
        np.testing.assert_almost_equal(
            results['map'], (self.ap0 + self.ap1) / 2)


class TestEvalDetectionVOCDifficults(unittest.TestCase):

    minoverlap = 0.5
    rec = np.array([0., 0., 1.])
    prec = np.array([0., 0., 1. / 3.])

    def test_eval_detection_voc_difficult(self):
        bboxes = [np.array([
            [0., 0., 1., 1.], [0., 0., 2., 2.], [0.3, 0.3, 0.5, 0.5]])]
        labels = [np.array([0, 0, 0])]
        confs = [np.array([0.8, 0.9, 1.])]
        gt_bboxes = [np.array([[0., 0., 1., 0.9], [1., 1., 2., 2.]])]
        gt_labels = [np.array([0, 0])]
        gt_difficults = [np.array([False, True])]
        # iou is [0.95, 0.422, 0.3789] and [0.142, 0.444, 0.048]

        results = eval_detection_voc(
            bboxes, labels, confs, gt_bboxes, gt_labels,
            n_class=1, gt_difficults=gt_difficults, minoverlap=self.minoverlap)
        np.testing.assert_equal(results[0]['recall'], self.rec)
        np.testing.assert_equal(results[0]['precision'], self.prec)