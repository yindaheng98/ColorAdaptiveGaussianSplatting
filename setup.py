#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

package_dir = {
    'cags': 'cags',
}
packages = ['cags'] + ["cags." + package for package in find_packages(where="cags")]

setup(
    name='cags',
    version='0.12.0',
    author='yindaheng98',
    author_email='yindaheng98@gmail.com',
    url='https://github.com/yindaheng98/cags',
    description=u'CAGS: Color-Adaptive 3D Gaussian Splatting',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=packages,
    package_dir={'cags': 'cags'},
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
        'opencv-python',
    ],
)
