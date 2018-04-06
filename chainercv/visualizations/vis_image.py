import numpy as np


def vis_image(img, ax=None):
    """Visualize a color image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)` or
            :math:`(1, height, width)`.
            This is an RGB or greryscale image and the range of its value is
            :math:`[0, 255]`. If this is :obj:`None`, no image is displayed.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    from matplotlib import pyplot as plot
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    if img is not None:
        # CHW -> HWC
        img = img.transpose((1, 2, 0)).astype(np.uint8)
        if img.shape[2] == 1:
            img = img[:, :, 0]
        ax.imshow(img)
    return ax
