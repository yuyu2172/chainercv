import argparse
import numpy as np
import os

import chainer
from chainer import training
from chainer.training import extensions

from chainercv.datasets import CamVidDataset
from chainercv.datasets.camvid.camvid_dataset import camvid_label_names
from chainercv.datasets import TransformDataset
from chainercv.extensions import SemanticSegmentationVisReport
from chainercv import transforms

from segnet import SegNet
from segnet import SegNetLoss



class_weight = np.array([
    0.58872014284134, 0.51052379608154, 2.6966278553009, 0.45021694898605,
    1.1785038709641, 0.77028578519821, 2.4782588481903, 2.5273461341858,
    1.0122526884079, 3.2375309467316,
    4.1312313079834, 0], dtype=np.float32)

class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV Semantic Segmentation example with FCN')
    parser.add_argument('--batch_size', '-b', type=int, default=4,
                        help='Number of images in each mini-batch')
    parser.add_argument('--iteration', '-i', type=int, default=10000,
                        help='Number of iteration to carry out')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--lr', '-l', type=float, default=0.1,
                        help='Learning rate of the optimizer')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batch_size))
    print('# iteration: {}'.format(args.iteration))
    print('')
    batch_size = args.batch_size
    iteration = args.iteration
    gpu = args.gpu
    lr = args.lr
    out = args.out
    resume = args.resume

    # prepare datasets
    def transform(in_data):
        img, label = in_data
        vgg_subtract_bgr = np.array(
            [103.939, 116.779, 123.68], np.float32)[:, None, None]
        img -= vgg_subtract_bgr
        return img, label

    # train_data = VOCSemanticSegmentationDataset(mode='train')
    train_data = CamVidDataset(mode='train')
    val_data = CamVidDataset(mode='val')
    test_data = CamVidDataset(mode='test')
    train_data = TransformDataset(train_data, transform)
    val_data = TransformDataset(val_data, transform)
    test_data = TransformDataset(test_data, transform)

    # set up FCN32s
    n_class = len(camvid_label_names)
    model = SegNet()
    model = SegNetLoss(model, class_weight=class_weight, train_depth=4)
    if gpu != -1:
        model.to_gpu(gpu)
        chainer.cuda.get_device(gpu).use()

    # prepare an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # prepare iterators
    train_iter = chainer.iterators.SerialIterator(
        train_data, batch_size=batch_size)
    val_iter = chainer.iterators.SerialIterator(
        val_data, batch_size=1, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (iteration, 'iteration'), out=out)

    trainer.extend(extensions.snapshot_object(
        model, 'snapshot_model'), trigger=(iteration, 'iteration'))

    val_interval = 5000, 'iteration'
    plot_interval = 1000, 'iteration'
    log_interval = 10, 'iteration'

    trainer.extend(
        TestModeEvaluator(val_iter, model, device=gpu), trigger=val_interval)

    # reporter related
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'main/time',
         'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy',
         'main/accuracy_cls', 'validation/main/accuracy_cls',
         'main/miu', 'validation/main/miu',
         'main/fwavacc', 'validation/main/fwavacc']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # visualize training
    trainer.extend(
        extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            trigger=plot_interval, file_name='loss.png')
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            trigger=plot_interval, file_name='accuracy.png')
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy_cls', 'validation/main/accuracy_cls'],
            trigger=plot_interval, file_name='accuracy_cls.png')
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/miu', 'validation/main/miu'],
            trigger=plot_interval, file_name='iu.png')
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/fwavacc', 'validation/main/fwavacc'],
            trigger=plot_interval, file_name='fwavacc.png')
    )

    # def vis_transform(in_data):
    #     vgg_subtract_bgr = np.array(
    #         [103.939, 116.779, 123.68], np.float32)[:, None, None]
    #     img, label = in_data
    #     img += vgg_subtract_bgr
    #     img, label = transforms.chw_to_pil_image((img, label))
    #     return img, label

    # trainer.extend(
    #     SemanticSegmentationVisReport(
    #         range(10),  # visualize outputs for the first 10 data of test_data
    #         test_data,
    #         model,
    #         n_class=n_class,
    #         predict_func=model.predict,  # a function to predict output
    #         vis_transform=vis_transform
    #     ),
    #     trigger=val_interval, invoke_before_training=True)

    trainer.extend(extensions.dump_graph('main/loss'))

    if resume:
        chainer.serializers.load_npz(os.path.expanduser(resume), trainer)

    trainer.run()


if __name__ == '__main__':
    main()
