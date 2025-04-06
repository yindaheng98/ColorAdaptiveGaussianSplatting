# CAGS: Color-Adaptive 3D Gaussian Splatting

## Prerequisites

* [Pytorch](https://pytorch.org/) (v2.4 or higher recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive) (12.4 recommended, should match with PyTorch version)

## Install

```sh
pip install --upgrade git+https://github.com/yindaheng98/ScalableVQ.git@master
pip install --upgrade git+https://github.com/yindaheng98/PostRenderPerspectiveAlign.git@master
pip install --upgrade git+https://github.com/yindaheng98/ColorAdaptiveGaussianSplatting.git@main
```
If you have trouble with [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting) or [`reduced-3dgs`](https://github.com/yindaheng98/reduced-3dgs), you can install them from source:
```sh
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
pip install --upgrade git+https://github.com/yindaheng98/reduced-3dgs.git@main
```

## Install for Development

```shell
git clone --recursive https://github.com/yindaheng98/ColorAdaptiveGaussianSplatting
cd ColorAdaptiveGaussianSplatting
pip install tqdm plyfile scikit-learn numpy opencv-python
pip install --upgrade --target . --no-deps git+https://github.com/yindaheng98/ScalableVQ.git@master
pip install --upgrade --target . --no-deps git+https://github.com/yindaheng98/PostRenderPerspectiveAlign.git@master
pip install --upgrade --target . --no-deps git+https://github.com/yindaheng98/gaussian-splatting.git@master
pip install --upgrade --target . --no-deps git+https://github.com/yindaheng98/reduced-3dgs.git@main
pip install --upgrade --target . --no-deps .
```

## Build Draco

```sh
mkdir build && cd build && cmake ../submodules/draco -DCMAKE_BUILD_TYPE=Release && cd ../
cmake --build build --config Release --target draco_encoder
cmake --build build --config Release --target draco_decoder
```