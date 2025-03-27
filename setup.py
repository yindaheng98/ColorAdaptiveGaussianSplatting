#!/usr/bin/env python
# coding: utf-8

from setuptools import setup

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

package_dir = {
    'cags': 'cags',
}

setup(
    name='cags',
    version='0.2',
    author='yindaheng98',
    author_email='yindaheng98@gmail.com',
    url='https://github.com/yindaheng98/cags',
    description=u'CAGS: Color-Adaptive 3D Gaussian Splatting',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir=package_dir,
    packages=[key for key in package_dir],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'tqdm',
        'plyfile',
        'scikit-learn',
        'torch',
        'torchvision',
        'numpy',
        'gaussian-splatting',
        'reduced-3dgs',
    ],
)
