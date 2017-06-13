import numpy as np
import os

from chainercv.datasets import VisionDataset


def find_label_names(directory):
    label_names = [d for d in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, d))]
    label_names.sort()
    return label_names


def _ends_with_img_ext(filename):
    img_extensions = [
        '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
        '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in img_extensions)


def parse_classification_dataset(root, label_names,
                                 check_img_file=_ends_with_img_ext):
    img_paths = []
    labels = []
    for label_name in os.listdir(root):
        label_dir = os.path.join(root, label_name)
        if not os.path.isdir(label_dir):
            continue

        for cur_dir, _, filenames in sorted(os.walk(label_dir)):
            for filename in filenames:
                if check_img_file(filename):
                    img_paths.append(os.path.join(cur_dir, filename))
                    labels.append(label_names.index(label_name))

    return img_paths, np.array(labels, np.int32)


class FolderDataset(VisionDataset):

    def __init__(self, root, check_img_file=None):
        label_names = find_label_names(root)
        if check_img_file is None:
            check_img_file = _ends_with_img_ext

        img_paths, labels = parse_classification_dataset(
            root, label_names, check_img_file)
        super(ImageFolderDataset, self).__init__(
            img_paths, labels)
