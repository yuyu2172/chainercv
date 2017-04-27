import numpy as np
import os

import chainer

from chainer.dataset import download

from chainercv import utils


root = 'pfnet/chainercv/camvid'
camvid_label_names = ['Sky', 'Building', 'Column-Pole', 'Road',
           'Sidewalk', 'Tree', 'Sign-Symbol', 'Fence', 'Car', 'Pedestrain',
           'Bicyclist', 'Void']


def _create_array_from_directory(base_path):
    def wrapped(path):
        modes = ['train', 'val', 'test']
        raw = {}
        for mode in modes:
            imgs = []
            labels = []
            dir_name = os.path.join(base_path, mode)
            label_dir = os.path.join(base_path, mode + 'annot')
            for fn in os.listdir(dir_name):
                print fn
                img = utils.read_image_as_array(
                    os.path.join(dir_name, fn), dtype=np.float32, force_color=True)
                imgs.append(img)
                label = utils.read_image_as_array(
                    os.path.join(label_dir, fn), dtype=np.int32, force_color=False)
                labels.append(label)
            imgs = np.stack(imgs)
            labels = np.stack(labels)
            raw[mode + '_img'] = imgs
            raw[mode + '_label'] = labels
        np.savez_compressed(
            path,
            train_img=raw['train_img'],
            train_label=raw['train_label'],
            val_img=raw['val_img'],
            val_label=raw['val_label'],
            test_img=raw['test_img'],
            test_label=raw['test_label']
        )
        return raw
    return wrapped


class CamVidDataset(chainer.datasets.TupleDataset):

    def __init__(self, mode='train', base_path=None):
        data_root = download.get_dataset_directory(root)
        npz_fn = os.path.join(data_root, 'camvid.npz')
        raw = download.cache_or_load_file(
            npz_fn,
            _create_array_from_directory(base_path),
            np.load
        )

        super(CamVidDataset, self).__init__(
            raw[mode + '_img'], raw[mode + '_label'])


if __name__ == '__main__':
    base_path = '/home/leus/playground/SegNet-Tutorial/CamVid/'
    dataset = CamVidDataset(base_path=base_path)
