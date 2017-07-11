import numpy as np
import six

import chainer
from chainer.dataset.convert import to_device
from chainer.dataset.iterator import Iterator


class TransformIterator(Iterator):

    """
    Usage:
        
        transform that uses
        1. GPU
        2. Batch processing
        3. History of samples

    """

    def __init__(self, iterator, transform, device=None):
        self._iterator = iterator
        self.transform = transform
        self.device = device

    def __next__(self):
        batch = self._iterator.next()

        if len(batch) == 0:
            raise ValueError('batch is empty')

        first_elem = batch[0]
        import cupy

        with cupy.cuda.Device(self.device):
            if isinstance(first_elem, tuple):
                in_arrays = []
                for i in six.moves.range(len(first_elem)):
                    in_arrays.append(
                        _batch_to_device([example[i] for example in batch],
                                        self.device))
                in_arrays = self.transform(tuple(in_arrays))
                batch = zip(*in_arrays)
                return batch

            elif isinstance(first_elem, dict):
                in_arrays = {}

                for key in first_elem:
                    in_arrays[key] = _batch_to_device(
                        [example[key] for example in batch], self.device)
                in_arrays = self.transform(in_arrays)
                batch = [[in_arrays[key][i] for key in in_arrays.keys()]
                        for i in range(len(batch))]
                return batch

            else:
                batch = self.transform(_batch_to_device(batch, self.device))
                return batch

    def finalize(self):
        return self._iterator.finalize()

    def serialize(self, serializer):
        return self._iterator.serialize()

    def __getattr__(self, name):
        if hasattr(self._iterator, name):
            return getattr(self._iterator, name)
        else:
            raise AttributeError


def _batch_to_device(xs, device):
    """Batch of arrays (e.g. list of arrays or batch array)

    It is designed to have minimal amount of malloc call.

    """
    # TODO: exist if device is CPU
    import cupy

    shapes = [x.shape for x in xs]
    elem_sizes = [np.prod(x.shape) for x in xs]
    array_split = np.cumsum(elem_sizes)

    xs_gpu = cupy.empty((np.sum(elem_sizes, dtype=np.int32),), dtype=xs[0].dtype)
    xs_gpu = cupy.split(xs_gpu, array_split)
    xs_gpu = [x_gpu.reshape(shape) for x_gpu, shape in zip(xs_gpu, shapes)]

    for x, x_gpu in zip(xs, xs_gpu):
        x_gpu.set(np.array(x, copy=False))

    return xs_gpu
