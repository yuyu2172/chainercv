import numpy as np

import chainer
from chainercv.datasets import CamVidDataset
from chainercv.datasets.camvid.camvid_dataset import camvid_label_names
from segnet import SegNet
from segnet import SegNetLoss
from chainercv.datasets import TransformDataset
from chainercv.evaluations import eval_semantic_segmentation

from chainer.dataset.convert import concat_examples


class_weight = np.array([
    0.58872014284134, 0.51052379608154, 2.6966278553009, 0.45021694898605,
    1.1785038709641, 0.77028578519821, 2.4782588481903, 2.5273461341858,
    1.0122526884079, 3.2375309467316,
    4.1312313079834, 0], dtype=np.float32)


if __name__ == '__main__':
    # prepare datasets
    def transform(in_data):
        img, label = in_data
        vgg_subtract_bgr = np.array(
            [103.939, 116.779, 123.68], np.float32)[:, None, None]
        img -= vgg_subtract_bgr
        return img, label
    test_data = CamVidDataset(mode='test')
    test_data = TransformDataset(test_data, transform)
    device = 0
    model = SegNet()
    model.train = False
    segnet_loss = SegNetLoss(model, class_weight, train_depth=4)
    chainer.serializers.load_npz('result/snapshot_model', segnet_loss)
    model.to_gpu(device)

    acc = []
    acc_cls = []
    miu = []
    fwavacc = []
    for i in range(len(test_data)):
        batch = test_data[i:i+1]
        in_arrs = concat_examples(batch, device=device)
        img = in_arrs[0]
        label = in_arrs[1]
        y = model(img, depth=4)

        y = chainer.cuda.to_cpu(y.data)
        label = chainer.cuda.to_cpu(label)
        y = np.argmax(y, axis=1)[:, None]
        score = eval_semantic_segmentation(y, label, model.n_class)
        acc.append(score[0])
        acc_cls.append(score[1])
        miu.append(score[2])
        fwavacc.append(score[3])
    print np.mean(acc)
    print np.mean(acc_cls)
    print np.mean(miu)
    print np.mean(fwavacc)
