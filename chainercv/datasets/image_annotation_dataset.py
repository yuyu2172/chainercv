import numpy as np
import os

import chainer

from chainercv.utils import read_image


class ImageAnnotationDataset(chainer.dataset.DatasetMixin):

    def __init__(self, img_paths, *annos):
        self.img_paths = img_paths
        self.annos = annos

        for anno in self.annos:
            if len(anno) != len(self.img_paths):
                raise ValueError(
                    'the number of annotations and image paths '
                    'need to be same')

    def __len__(self):
        return len(self.img_paths)

    def get_example(self, i):
        img = read_image(self.img_paths[i])
        out = [img]
        for anno in self.annos:
            out.append(anno[i])
        return tuple(out)
