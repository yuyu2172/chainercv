import matplotlib
matplotlib.use('agg')
import argparse
import numpy as np

import chainer
from chainer.datasets import split_dataset_n_random
from chainer.datasets import TransformDataset
from chainer import iterators
from chainer.links import Classifier
from chainer import training
from chainer.training import extensions

from chainercv.datasets import DirectoryParsingClassificationDataset
from chainercv.iterators import TransformIterator

from chainercv.transforms import center_crop
from chainercv.transforms import pca_lighting
from chainercv.transforms import random_crop
from chainercv.transforms import random_flip
from chainercv.transforms import scale

from chainercv.links import ResNet50

from chainercv.links.model.resnet.resnet import _imagenet_mean


class IteratorTransform(object):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        imgs, labels = in_data
        out_imgs = []
        xp = chainer.cuda.get_array_module(imgs)

        mean = xp.array(self.mean)
        for img in imgs:
            scale_size = np.random.randint(256, 481)
            img = scale(img, scale_size)
            img = random_flip(img, x_random=True)
            img = random_crop(img, (224, 224))
            img = pca_lighting(img, 22.5)
            img -= mean
            out_imgs.append(img)
        return out_imgs, labels


def val_transform(in_data):
    img, label = in_data
    img = scale(img, 256)
    img = center_crop(img, (224, 224))
    img -= _imagenet_mean
    return img, label


def get_train_iter(train_data, batchsize, devices,
                   iterator_transform=None, loaderjob=None):
    if len(devices) > 1:
        assert batchsize % len(devices) == 0
        per_device_batchsize = batchsize // len(devices)
        train_iter = [
            chainer.iterators.MultiprocessIterator(
                i, per_device_batchsize,
                n_processes=loaderjob, shared_mem=10000000)
            for i in split_dataset_n_random(train_data, len(devices))]
        if iterator_transform is not None:
            train_iter = [TransformIterator(it, iterator_transform, device)
                          for it, device in zip(train_iter, devices)]
    else:
        train_iter = chainer.iterators.MultiprocessIterator(
            train_data, batch_size=batchsize,
            n_processes=loaderjob, shared_mem=100000000)
        if iterator_transform is not None:
            train_iter = TransformIterator(
                train_iter, iterator_transform, devices[0])
    return train_iter


def get_updater(train_iter, optimizer, devices):
    if len(devices) > 1:
        updater = chainer.training.updaters.MultiprocessParallelUpdater(
            train_iter, optimizer, devices=devices)
    else:
        updater = chainer.training.updater.StandardUpdater(
            train_iter, optimizer, device=devices[0])
    return updater


def main():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to root of the train dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument('--pretrained_model')
    parser.add_argument('--gpus', type=int, nargs="*", default=[-1])
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=90)
    args = parser.parse_args()

    import os
    import pickle
    cache_fn = 'cache_train_dataset.pkl'
    if os.path.exists(cache_fn):
        with open(cache_fn, 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = DirectoryParsingClassificationDataset(args.train)
        with open(cache_fn, 'wb') as f:
            pickle.dump(train_data, f, protocol=2)
    print('finished loading train')

    val_data = DirectoryParsingClassificationDataset(args.val)
    val_data = TransformDataset(val_data, val_transform)

    train_iter = get_train_iter(train_data, args.batchsize, args.gpus)
    val_iter = iterators.MultiprocessIterator(
        val_data, args.batchsize,
        repeat=False, shuffle=False, shared_mem=10000000)

    extractor = ResNet50(n_class=1000)
    model = Classifier(extractor)

    lr = args.lr / len(args.gpu)
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0001))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    updater = get_updater(train_iter, optimizer, args.gpus)

    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.step_size, 'epoch'))

    log_interval = 0.1, 'epoch'
    print_interval = 0.1, 'epoch'
    plot_interval = 1, 'epoch'

    trainer.extend(extensions.LogReport(trigger=log_interval))

    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss']
    ), trigger=print_interval)

    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )
    trainer.extend(
        extensions.Evaluator(val_iter, model),
        trigger=(1, 'epoch')
    )

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
