# CAGS: Color-Adaptive 3D Gaussian Splatting

This repo is the **Python implementation of CAGS**, a package for scalable compression and reconstruction of 3D Gaussian Splatting point clouds. It is built on top of [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting), [`reduced-3dgs`](https://github.com/yindaheng98/reduced-3dgs), [`ScalableVQ`](https://github.com/yindaheng98/ScalableVQ), and Google's Draco geometry codec.

CAGS converts Gaussian attributes into scalable base and enhancement layers, optionally compresses the base layer with Draco, splits large point clouds into spatial tiles, and stores only changed Gaussians for later frames in a dynamic sequence. The package provides command-line tools for single-frame quantization, tiled quantization, inter-frame encoding / decoding, progressive layer pickup, and rendering reconstructed results.

## Features

* [x] Standard Python package with `pip install` support
* [x] Scalable vector quantization for 3DGS attributes: position, rotation, opacity, scaling, DC colour and higher-order SH features
* [x] Base-layer / enhancement-layer layout for progressive quality reconstruction
* [x] Optional Draco compression for compact base-layer geometry storage
* [x] Morton-order tiling and tile stitching for large Gaussian point clouds
* [x] Inter-frame coding modes for dynamic sequences: full frame, quantized-difference and attribute-threshold difference
* [x] Standalone CLI modules for `quantize`, `tile`, `encode`, `decode`, `pickup`, and `render`

## Install

### Prerequisites

* Python >= 3.10
* [PyTorch](https://pytorch.org/) (>= v2.4 recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive) (12.4 recommended, match with PyTorch version)
* [CMake](https://cmake.org/) and a C++ compiler for building bundled Draco tools
* [`ScalableVQ`](https://github.com/yindaheng98/ScalableVQ)
* [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting)
* [`reduced-3dgs`](https://github.com/yindaheng98/reduced-3dgs)

Install dependencies that are not declared by the package metadata:

```shell
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/ScalableVQ.git@master
```

(Optional) If you have trouble with [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting) or [`reduced-3dgs`](https://github.com/yindaheng98/reduced-3dgs), install them from source:

```shell
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master --no-build-isolation
pip install --upgrade git+https://github.com/yindaheng98/reduced-3dgs.git@main --no-build-isolation
```

## PyPI Install

```shell
pip install --upgrade ColorAdaptiveGaussianSplatting
```

or build the latest version from source:

```shell
pip install wheel setuptools
pip install --upgrade git+https://github.com/yindaheng98/ColorAdaptiveGaussianSplatting.git@main --no-build-isolation
```

### Development Install

```shell
git clone --recursive https://github.com/yindaheng98/ColorAdaptiveGaussianSplatting.git
cd ColorAdaptiveGaussianSplatting
pip install tqdm plyfile scikit-learn numpy opencv-python
pip install --upgrade --target . --no-deps git+https://github.com/yindaheng98/ScalableVQ.git@master
pip install --upgrade --target . --no-deps git+https://github.com/yindaheng98/gaussian-splatting.git@master
pip install --upgrade --target . --no-deps git+https://github.com/yindaheng98/reduced-3dgs.git@main
pip install --upgrade --target . --no-deps .
```

The Draco encoder and decoder are built from the `submodules/draco` submodule during installation. Use `git clone --recursive` or run `git submodule update --init --recursive` before local builds.

## Data Layout

CAGS reads and writes the same directory layout used by Gaussian Splatting:

```text
scene_or_frame/
|-- cfg_args
|-- cameras.json
`-- point_cloud/
    `-- iteration_30000/
        `-- point_cloud.ply
```

For sequence coding, the initial frame is passed separately with `--source_init`, while the remaining frames are addressed by `--source`, `--frame_format`, `--frame_start`, and `--frame_end`.

```text
sequence/
|-- frame0/
|   `-- point_cloud/iteration_30000/point_cloud.ply
|-- frame1/
|   `-- point_cloud/iteration_30000/point_cloud.ply
`-- frame2/
    `-- point_cloud/iteration_30000/point_cloud.ply
```

## Command-Line Usage

### Quantize a Single Frame

```shell
python -m cags.quantize \
    -s output/frame0 \
    -d compressed/frame0 \
    -i 30000 \
    --draco \
    -o "n_bit_baselayer=4" \
    -o "n_bits_proposal=[2,2,2,2]"
```

This writes `point_cloud_quantized.ply` plus codebook and enhancement-layer sidecar files under `compressed/frame0/point_cloud/iteration_30000/`.

### Dequantize a Single Frame

```shell
python -m cags.quantize \
    -s compressed/frame0 \
    -d compressed/frame0 \
    -i 30000 \
    --draco \
    --dequantize
```

Dequantization loads `point_cloud_quantized.ply` from the destination directory and writes a reconstructed `point_cloud.ply` next to it.

### Tile and Stitch a Frame

```shell
python -m cags.tile \
    -s output/frame0 \
    -d compressed_tiled/frame0 \
    -i 30000 \
    --draco \
    -o "n_bit_baselayer=4" \
    -o "n_bits_proposal=[2,2,2,2]"
```

To reconstruct tiled output:

```shell
python -m cags.tile \
    -s compressed_tiled/frame0 \
    -d compressed_tiled/frame0 \
    -i 30000 \
    --draco \
    --stitching
```

Tiling uses `MortonTiling` by default, stores shared codebooks beside the base layer, and writes per-tile quantized files in a `*_tiles/` directory.

### Encode a Dynamic Sequence

```shell
python -m cags.encode \
    --source_init output/frame0 \
    --destination_init compressed/frame0 \
    --iteration_init 30000 \
    -s output/sequence \
    -d compressed/sequence \
    -i 30000 \
    --frame_format "frame%d" \
    --frame_start 1 \
    --frame_end 30 \
    --interframe interframe \
    --draco \
    -o "n_bit_baselayer=4" \
    -o "n_bits_proposal=[2,2,2,2]"
```

The first frame initializes the codec state. Later frames are encoded as differences from the previous reconstructed frame and store a packed `.mask.npz` file indicating which Gaussians changed.

Available inter-frame modes:

* `none`: encode every Gaussian in every frame
* `quantize`: compare quantized attribute ids and encode changed Gaussians
* `interframe`: compare attributes with configurable thresholds and encode changed Gaussians

If you disable first-frame tiling with `--no_tiling_first`, also pass `--no_tiling_rest`.

### Decode a Dynamic Sequence

```shell
python -m cags.decode \
    --source_init compressed/frame0 \
    --destination_init reconstructed/frame0 \
    --iteration_init 30000 \
    -s compressed/sequence \
    -d reconstructed/sequence \
    -i 30000 \
    --frame_format "frame%d" \
    --frame_start 1 \
    --frame_end 30 \
    --interframe interframe \
    --draco
```

The decoder reconstructs a full `point_cloud.ply` for the initial frame and every requested sequence frame.

### Pick Up Progressive Layers

```shell
python -m cags.pickup \
    -s compressed/frame0 \
    -d preview/frame0 \
    -i 30000 \
    --draco \
    --pickup_sh_degree 1 \
    -l rotation_re=0 \
    -l rotation_im=0 \
    -l opacity=1 \
    -l scaling=1 \
    -l features_dc=1 \
    -l features_rest_0=0
```

`pickup` copies only selected enhancement layers and immediately reconstructs a preview `point_cloud.ply`. The layer dictionary must include every attribute key required by the selected SH degree.

### Render Reconstructed Results

```shell
python -m cags.render \
    -s data/frame0 \
    -d reconstructed/frame0 \
    -i 30000 \
    --mode base \
    --device cuda
```

Rendered RGB images, ground truth images, depth previews, depth arrays, and camera JSON files are saved under `reconstructed/frame0/ours_30000/`.

## API Usage

### Codec

```python
from gaussian_splatting import GaussianModel
from cags.encode import prepare_codec

codec = prepare_codec(
    draco=True,
    tiling_first=True,
    tiling_rest=True,
    interframe="interframe",
    n_bit_baselayer=4,
    n_bits_proposal=[2, 2, 2, 2],
)

frame0 = GaussianModel(3).to("cuda")
frame0.load_ply("output/frame0/point_cloud/iteration_30000/point_cloud.ply")
codec.encode_init(frame0, "compressed/frame0/point_cloud/iteration_30000/point_cloud.ply")

frame1 = GaussianModel(3).to("cuda")
frame1.load_ply("output/frame1/point_cloud/iteration_30000/point_cloud.ply")
codec.encode_next(frame1, "compressed/frame1/point_cloud/iteration_30000/point_cloud.ply")
```

`Codec` keeps state between frames: the initial frame defines codebooks, tile order and inter-frame reference data; each following frame is encoded against the previous reconstructed frame.

### Decode With the Codec

```python
from gaussian_splatting import GaussianModel
from cags.encode import prepare_codec

codec = prepare_codec(
    draco=True,
    tiling_first=True,
    tiling_rest=True,
    interframe="interframe",
)

frame0 = codec.decode_init(
    GaussianModel(3).to("cuda"),
    "compressed/frame0/point_cloud/iteration_30000/point_cloud.ply",
)
frame0.save_ply("reconstructed/frame0/point_cloud/iteration_30000/point_cloud.ply")

frame1 = codec.decode_next(
    GaussianModel(3).to("cuda"),
    "compressed/frame1/point_cloud/iteration_30000/point_cloud.ply",
)
frame1.save_ply("reconstructed/frame1/point_cloud/iteration_30000/point_cloud.ply")
```

### Direct Quantizer Usage

```python
from gaussian_splatting import GaussianModel
from cags.quantization import DracoCompressedScalableQuantizer

gaussians = GaussianModel(3).to("cuda")
gaussians.load_ply("output/frame0/point_cloud/iteration_30000/point_cloud.ply")

quantizer = DracoCompressedScalableQuantizer(
    n_bit_baselayer=4,
    n_bits_proposal=[2, 2, 2, 2],
)
quantizer.save_quantized(
    gaussians,
    "compressed/frame0/point_cloud/iteration_30000/point_cloud_quantized.ply",
)

reconstructed = quantizer.load_quantized(
    GaussianModel(3).to("cuda"),
    "compressed/frame0/point_cloud/iteration_30000/point_cloud_quantized.ply",
)
```

## Design: Scalable 3DGS Coding

The core abstraction separates **attribute quantization**, **spatial tiling**, and **inter-frame extraction** so they can be composed by `Codec`.

### Scalable Quantizer

`ScalableQuantizer` extends the reduced-3DGS quantizer by converting each Gaussian attribute into a base layer and zero or more enhancement layers:

```text
Gaussian attributes -> Vector quantization -> Base layer + enhancement layers
```

The base layer stores coarse ids and codebooks. Enhancement layers refine the same attributes progressively, which enables lower-bitrate preview reconstruction through `pickup`.

### Draco-Compressed Base Layer

`DracoCompressedScalableQuantizer` stores base-layer codes in a temporary PLY, compresses them into `.drc` with the bundled Draco encoder, and restores them with the bundled decoder during loading. Enhancement layers remain `.npz` sidecar files.

### Tiling

`TillingScalableQuantizer` wraps any scalable quantizer with a tiling strategy:

```text
Gaussian point cloud -> Morton / average split tiles -> Per-tile quantization -> Stitching
```

`MortonTiling` sorts Gaussians by Morton code before splitting, keeping nearby points in the same tile. `AverageSplitTiling` is also used for changed Gaussians in inter-frame residuals.

### Inter-Frame Extraction

The codec first encodes an initialization frame. For each later frame, an inter-frame extractor selects changed Gaussians and stores a packed mask:

```text
Previous frame + current frame -> Difference mask -> Quantized changed Gaussians
```

`NoInterframeExtractor` always stores full frames, `QuantizedInterframeExtractor` compares quantized ids, and `InterframeExtractor` compares positions, rotations, opacity, scaling and SH features with configurable thresholds.

## Extending: Adding a Custom Component

CAGS components are small Python classes. To add a new spatial partitioner, implement `AbstractTiling.produce_tiling` and pass it to `TillingScalableQuantizer`. To add a new inter-frame policy, implement `AbstractInterframeExtractor.diff_mask` and pass it to `Codec`.

```python
from cags.codec import Codec
from cags.interframe import AbstractInterframeExtractor
from cags.quantization import ScalableQuantizer
from cags.tilequant import TillingScalableQuantizer
from cags.tiling import MortonTiling

class MyInterframeExtractor(AbstractInterframeExtractor):
    def diff_mask(self, frame):
        # Return a bool tensor with one entry per Gaussian.
        ...

codec = Codec(
    frame_extractor=MyInterframeExtractor(),
    frame_quantizer=TillingScalableQuantizer(
        ScalableQuantizer(n_bit_baselayer=4),
        MortonTiling(),
    ),
    tiling_first=True,
)
```

## Acknowledgement

This repo is developed based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [gaussian-splatting (packaged)](https://github.com/yindaheng98/gaussian-splatting), [reduced-3dgs](https://github.com/yindaheng98/reduced-3dgs), [ScalableVQ](https://github.com/yindaheng98/ScalableVQ), and [Draco](https://github.com/google/draco). Many thanks to the authors for open-sourcing their codebases.
