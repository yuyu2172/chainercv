#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages
import os
import pkg_resources

# from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy as np


description = """
Collection of Deep Learning Computer Vision Algorithms implemented in Chainer
"""

ext_modules = [
    Extension('chainercv.utils.bbox._nms_gpu_post',
              ['chainercv/utils/bbox/_nms_gpu_post.pyx']),
]
# cmdclass = {'build_ext': build_ext}

install_requires = [
    'chainer==2.0',
    'Cython',
    'Pillow'
]


from distutils.command.build_ext import build_ext as _build_ext
try:
    from Cython.Distutils import build_ext as _build_ext
    # from Cython.Distutils import Extension # to get pyrex debugging symbols
    cython = True
except ImportError:
    cython = False

class build_ext(_build_ext):
    # https://github.com/pandas-dev/pandas/blob/f18378dc5af59c137e3579ff83806296c2222321/setup.py#L371
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if hasattr(ext, 'include_dirs') and not numpy_incl in ext.include_dirs:
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


class CheckingBuildExt(build_ext):
    """Subclass build_ext to get clearer report if Cython is necessary."""

    def check_cython_extensions(self, extensions):
        for ext in extensions:
            for src in ext.sources:
                if not os.path.exists(src):
                    raise Exception("""Cython-generated file '{}' not found.
                Cython is required to compile chainercv from a development branch.
                Please install Cython or download a release package of pandas.
                """.format(src))

    def build_extensions(self):
        self.check_cython_extensions(self.extensions)
        build_ext.build_extensions(self)

cmdclass = {}

cmdclass['build_ext'] = CheckingBuildExt


setup(
    name='chainercv',
    version='0.5.1',
    packages=find_packages(),
    author='Yusuke Niitani',
    author_email='yuyuniitani@gmail.com',
    license='MIT',
    description=description,
    install_requires=install_requires,
    include_package_data=True,
    # for Cython
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_dirs=[np.get_include()],
)
