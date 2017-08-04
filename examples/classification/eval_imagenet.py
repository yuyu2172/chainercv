import argparse
import sys
import time

import numpy as np

import chainer
import chainer.functions as F
from chainer import iterators

from chainercv.datasets import DirectoryParsingClassificationDataset
from chainercv.links import FeatureExtractionPredictor
from chainercv.links import InceptionV1
from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50
from chainercv.links import VGG16

from chainercv.utils import apply_prediction_to_iterator


class ProgressHook(object):

    def __init__(self, n_total):
        self.n_total = n_total
        self.start = time.time()
        self.n_processed = 0

    def __call__(self, imgs, pred_values, gt_values):
        self.n_processed += len(imgs)
        fps = self.n_processed / (time.time() - self.start)
        sys.stdout.write(
            '\r{:d} of {:d} images, {:.2f} FPS'.format(
                self.n_processed, self.n_total, fps))
        sys.stdout.flush()


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument(
        '--model', choices=(
            'vgg16', 'resnet50', 'resnet101', 'resnet152', 'inception_v1'))
    parser.add_argument('--pretrained_model')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--do_ten_crop', action='store_true')
    args = parser.parse_args()

    dataset = DirectoryParsingClassificationDataset(args.val)
    iterator = iterators.MultiprocessIterator(
        dataset, args.batchsize, repeat=False, shuffle=False,
        n_processes=6, shared_mem=300000000)

    if args.pretrained_model:
        pretrained_model = args.pretrained_model
    else:
        pretrained_model = 'imagenet'

    if args.model == 'vgg16':
        model = VGG16(pretrained_model=pretrained_model)
    elif args.model == 'resnet50':
        model = ResNet50(pretrained_model=pretrained_model)
    elif args.model == 'resnet101':
        model = ResNet101(pretrained_model=pretrained_model)
    elif args.model == 'resnet152':
        model = ResNet152(pretrained_model=pretrained_model)
    elif args.model == 'inception_v1':
        model = InceptionV1(pretrained_model=pretrained_model, n_class=1000)
    model = FeatureExtractionPredictor(model, do_ten_crop=args.do_ten_crop)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    print('Model has been prepared. Evaluation starts.')
    imgs, pred_values, gt_values = apply_prediction_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    del imgs

    pred_labels, = pred_values
    gt_labels, = gt_values

    accuracy = F.accuracy(
        np.array(list(pred_labels)), np.array(list(gt_labels))).data
    print()
    print('Top 1 Error {}'.format(1. - accuracy))


if __name__ == '__main__':
    main()
