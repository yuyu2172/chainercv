import numpy as np
import os

import chainer
from chainer.dataset import download

from chainercv import utils

try:
    from scipy.io import loadmat

    _available = True
except ImportError:
    _available = False


root = 'pfnet/chainercv/bsds500'
url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/' \
    'BSR/BSR_bsds500.tgz'


def _get_bsds_dataset():
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'BSR')
    if os.path.exists(base_path):
        # skip downloading
        return base_path

    download_file_path = utils.cached_download(url)
    ext = os.path.splitext(url)[1]
    utils.extractall(download_file_path, data_root, ext)
    return base_path


class BSDS500Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir='auto', split='train'):
        if not _available:
            raise ImportError(
                'scipy needs to be installed to use BSDS500Dataset')
        if data_dir == 'auto':
            data_dir = _get_bsds_dataset()
        self.data_dir = data_dir

        img_filenames = {}
        gt_filenames = {}
        for sp in ['train', 'val', 'test']:
            img_base_dir = os.path.join(
                self.data_dir, 'BSDS500/data/images/{}'.format(sp))
            gt_base_dir = os.path.join(
                self.data_dir, 'BSDS500/data/groundTruth/{}'.format(sp))
            img_filenames[sp] = [
                os.path.join(img_base_dir, filename)
                for filename in sorted(os.listdir(img_base_dir))
                if os.path.splitext(filename)[1] == '.jpg']
            gt_filenames[sp] = [
                os.path.join(gt_base_dir, filename)
                for filename in sorted(os.listdir(gt_base_dir))
                if os.path.splitext(filename)[1] == '.mat']

        if split == 'trainval':
            self.img_filenames = img_filenames['train'] + img_filenames['val']
            self.gt_filenames = gt_filenames['train'] + gt_filenames['val']
        elif split in ['train', 'val', 'test']:
            self.img_filenames = img_filenames[split]
            self.gt_filenames = gt_filenames[split]
        else:
            raise ValueError('split needs to be one of \'train\', \'val\', '
                             '\'test\', \'trainval\'')

    def __len__(self):
        return len(self.img_filenames)

    def get_example(self, i):
        img = utils.read_image(self.img_filenames[i], color=True)
        gt = loadmat(self.gt_filenames[i])
        regions = []
        contours = []
        for gt_sample in gt['groundTruth'][0]:
            regions.append(gt_sample[0, 0][0])
            contours.append(gt_sample[0, 0][1])
        return (img, np.stack(regions).astype(np.int32),
                np.stack(contours).astype(np.int32))


if __name__ == '__main__':
    from chainercv.visualizations import vis_image
    import matplotlib.pyplot as plt

    dataset = BSDS500Dataset()
    img, labels, contours = dataset[0]

    for i in range(len(labels)):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        vis_image(img, ax1)
        ax2.imshow(labels[i])
        ax3.imshow(contours[i])
        plt.show()
