import chainer
from chainer.links import GoogLeNet

from chainercv.links.model.inception.inception_v1 import InceptionV1 


def main():
    chainer_model = GoogLeNet(pretrained_model='auto')
    cv_model = InceptionV1(pretrained_model=None, n_class=1000)

    cv_model.conv1.conv.copyparams(chainer_model.conv1)
    # The pretrained weights are trained to accept BGR images.
    # Convert weights so that they accept RGB images.
    cv_model.conv1.conv.W.data[:] = cv_model.conv1.conv.W.data[:, ::-1]

    cv_model.conv2_reduce.conv.copyparams(chainer_model.conv2_reduce)
    cv_model.conv2.conv.copyparams(chainer_model.conv2)

    cv_model.inc3a.copyparams(chainer_model.inc3a)
    cv_model.inc3b.copyparams(chainer_model.inc3b)
    cv_model.inc4a.copyparams(chainer_model.inc4a)
    cv_model.inc4b.copyparams(chainer_model.inc4b)
    cv_model.inc4c.copyparams(chainer_model.inc4c)
    cv_model.inc4d.copyparams(chainer_model.inc4d)
    cv_model.inc4e.copyparams(chainer_model.inc4e)
    cv_model.inc5a.copyparams(chainer_model.inc5a)
    cv_model.inc5b.copyparams(chainer_model.inc5b)
    cv_model.fc.copyparams(chainer_model.loss3_fc)

    chainer.serializers.save_npz('inception_v1_from_caffe.npz', cv_model)


if __name__ == '__main__':
    main()
