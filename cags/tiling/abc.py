import abc
import copy
from typing import List, Tuple

import torch
import torch.nn as nn

from gaussian_splatting.gaussian_model import GaussianModel


class AbstractTiling(abc.ABC):
    @abc.abstractmethod
    def produce_tiling(self, model: GaussianModel) -> List[torch.Tensor]:
        raise NotImplementedError

    def pick_tile(self, model: GaussianModel, tile_gaussians_id: List[torch.Tensor]) -> List[GaussianModel]:
        tile = copy.deepcopy(model)
        tile._xyz = nn.Parameter(model._xyz[tile_gaussians_id])
        tile._features_dc = nn.Parameter(model._features_dc[tile_gaussians_id])
        tile._features_rest = nn.Parameter(model._features_rest[tile_gaussians_id])
        tile._opacity = nn.Parameter(model._opacity[tile_gaussians_id])
        tile._scaling = nn.Parameter(model._scaling[tile_gaussians_id])
        tile._rotation = nn.Parameter(model._rotation[tile_gaussians_id])
        return tile

    def pick_tiles(self, model: GaussianModel, tile_gaussians_ids: List[torch.Tensor]) -> List[GaussianModel]:
        return [self.pick_tile(model, tile_gaussians_id) for tile_gaussians_id in tile_gaussians_ids]

    def tiling(self, model: GaussianModel) -> Tuple[List[GaussianModel], List[torch.Tensor]]:
        tile_gaussians_ids = self.produce_tiling(model)
        return self.pick_tiles(model, tile_gaussians_ids), tile_gaussians_ids

    def stitching(self, models: List[GaussianModel]) -> GaussianModel:
        model = copy.deepcopy(models[0])
        model._xyz = nn.Parameter(torch.cat([m._xyz for m in models], dim=0))
        model._features_dc = nn.Parameter(torch.cat([m._features_dc for m in models], dim=0))
        model._features_rest = nn.Parameter(torch.cat([m._features_rest for m in models], dim=0))
        model._opacity = nn.Parameter(torch.cat([m._opacity for m in models], dim=0))
        model._scaling = nn.Parameter(torch.cat([m._scaling for m in models], dim=0))
        model._rotation = nn.Parameter(torch.cat([m._rotation for m in models], dim=0))
        return model

    def sort_as_tiles(self, model: GaussianModel, tile_gaussians_ids: List[torch.Tensor]) -> GaussianModel:
        return self.stitching(self.pick_tiles(model, tile_gaussians_ids))
