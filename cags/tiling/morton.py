from typing import List

import torch

from gaussian_splatting.gaussian_model import GaussianModel

from .abc import AbstractTiling


def __part1by2_64(n):
    n &= 0x1fffff                            # binary: 111111111111111111111,                                         len: 21
    n = (n | (n << 32)) & 0x1f00000000ffff   # binary: 11111000000000000000000000000000000001111111111111111,         len: 53
    n = (n | (n << 16)) & 0x1f0000ff0000ff   # binary: 11111000000000000000011111111000000000000000011111111,         len: 53
    n = (n | (n << 8)) & 0x100f00f00f00f00f  # binary: 1000000001111000000001111000000001111000000001111000000001111, len: 61
    n = (n | (n << 4)) & 0x10c30c30c30c30c3  # binary: 1000011000011000011000011000011000011000011000011000011000011, len: 61
    n = (n | (n << 2)) & 0x1249249249249249  # binary: 1001001001001001001001001001001001001001001001001001001001001, len: 61

    return n


def morton(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    morton_x, morton_y, morton_z = __part1by2_64(x), __part1by2_64(y), __part1by2_64(z)
    code = (morton_x << 0) | (morton_y << 1) | (morton_z << 2)
    return code


class MortonTiling(AbstractTiling):
    def __init__(self, n_gaussians_pre_tile: int = 8192):
        self.n_gaussians_pre_tile = n_gaussians_pre_tile

    def produce_tiling(self, model: GaussianModel) -> List[torch.Tensor]:
        int_xyz = model._xyz.detach().sort(dim=0).indices.sort(dim=0).indices.to(dtype=torch.int64)  # float point xyz to int xyz, keep the order
        morton_code = morton(int_xyz[..., 0], int_xyz[..., 1], int_xyz[..., 2])
        order = morton_code.argsort()
        tile_idx = torch.linspace(0, order.shape[0], order.shape[0] // self.n_gaussians_pre_tile).round().to(dtype=torch.int64)
        tile_idx[-1] = order.shape[0]
        tile_size = tile_idx[1:] - tile_idx[:-1]
        return torch.split(order, tile_size.tolist(), dim=0)
