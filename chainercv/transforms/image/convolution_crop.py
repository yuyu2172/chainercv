def convolution_crop(img, size, stride):
    _, H, W = img.shape

    hh = int(ceil((H - size[0]) / stride[0])) + 1
    ww = int(ceil((W - size[1]) / stride[1])) + 1

