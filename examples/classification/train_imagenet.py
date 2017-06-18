import argparse
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainer import iterators
from chainer.links import Classifier
from chainer.datasets import TransformDataset

from chainercv.datasets import DirectoryParsingClassificationDataset
from chainercv.links import ResNet50

from chainercv.transforms import scale
from chainercv.transforms import random_crop


_imagenet_mean = np.array(
    [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]


def transform(in_data):
    img, label = in_data
    img = scale(img, 256)
    img = random_crop(img, (224, 224))
    img -= _imagenet_mean
    return img, label


def main():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to root of the train dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument(
        '--model', choices=('resnet50'))
    parser.add_argument('--pretrained_model')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=90)
    args = parser.parse_args()

    train_data = DirectoryParsingClassificationDataset(args.train)
    train_data = TransformDataset(train_data, transform)
    val_data = DirectoryParsingClassificationDataset(args.val)

    train_iter = iterators.MultiprocessIterator(
        train_data, args.batchsize, repeat=False, shuffle=False,
        n_processes=6, shared_mem=300000000)
    val_iter = iterators.MultiprocessIterator(
        val_data, args.batchsize, repeat=False, shuffle=False,
        n_processes=6, shared_mem=300000000)

    if args.model == 'resnet50':
        model = ResNet50(pretrained_model=None, n_class=1000, feature='fc8')

    model = Classifier(model)

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0001))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

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

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
