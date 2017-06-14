from chainercv.datasets import ImageFolderDataset

import fire


def main(root):
    ImageFolderDataset(root=root)


if __name__ == '__main__':
    fire.Fire(main)
