import copy
import glob
import os
from typing import Dict, List, NamedTuple, Tuple
import torch
import tqdm
from gaussian_splatting.gaussian_model import GaussianModel
from cags.quantization import InterfaceScalableQuantizer
from cags.tiling import AbstractTiling
from scalablevq import Layer


class Tile(NamedTuple):
    layers_dict: Dict[str, List[Layer]]
    gaussians: GaussianModel
    xyz: torch.Tensor = None


class TillingScalableQuantizer:

    def __init__(self, quantizer: InterfaceScalableQuantizer, tiling: AbstractTiling):
        self.quantizer = quantizer
        self.tiling = tiling

    def quantize_tiling(self, model: GaussianModel, update=True, tile_gaussians_ids: List[torch.Tensor] = None) -> Tuple[Dict[str, List[Layer]], List[Tile], List[torch.Tensor]]:
        ids_dict, codebook_dict = self.quantizer.quantize(model, update_codebook=update)
        layers_dict = self.quantizer.layerize(model, ids_dict, codebook_dict, update_layers=update)
        raw_tiles, tile_ids = self.tiling.tiling(model, tile_gaussians_ids=tile_gaussians_ids)
        tiles = []
        for tile in tqdm.tqdm(raw_tiles, desc="Quantizing tiles"):
            ids_dict, codebook_dict = self.quantizer.quantize(tile, update_codebook=False)
            layers_dict = self.quantizer.layerize(tile, ids_dict, codebook_dict, update_layers=False)
            tiles.append(Tile(gaussians=tile, layers_dict=layers_dict))
        return layers_dict, tiles, tile_ids

    def dequantize_stitching(self, model: GaussianModel, tiles: List[Tile]) -> GaussianModel:
        for i in range(len(tiles)):
            ids_dict, codebook_dict = self.quantizer.delayerize(model.max_sh_degree, tiles[i].layers_dict)
            tile = self.quantizer.dequantize(tiles[i].gaussians, ids_dict, codebook_dict, xyz=tiles[i].xyz, replace=True)
            tiles[i] = tiles[i]._replace(gaussians=tile)
        return self.tiling.stitching([tile.gaussians for tile in tiles])

    def save_codebooks(self, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        self.quantizer.save_baselayer_codebook(ply_path, layers_dict)
        self.quantizer.save_enhencementlayers_codebook(ply_path, layers_dict)

    def save_tiles(self, ply_path: str, tiles: List[Tile]):
        tile_dir = os.path.splitext(ply_path)[0] + "_tiles"
        os.makedirs(tile_dir, exist_ok=True)
        for i, tile in enumerate(tqdm.tqdm(tiles, desc="Saving tiles")):
            tile_path = os.path.join(tile_dir, f"{i}.ply")
            self.quantizer.save_baselayer_codes(tile.gaussians, tile_path, tile.layers_dict)
            self.quantizer.save_enhencementlayers_codes(tile_path, tile.layers_dict)

    def save_quantized_tiles(self, model: GaussianModel, ply_path: str):
        layers_dict, tiles, _ = self.quantize_tiling(model, update=False)
        self.save_codebooks(ply_path, layers_dict)
        self.save_tiles(ply_path, tiles)

    def load_codebooks(self, max_sh_degree: int, ply_path: str, device) -> Dict[str, List[Layer]]:
        layers_dict = self.quantizer.load_baselayer_codebook(max_sh_degree, ply_path, device)
        layers_dict = self.quantizer.load_enhencementlayers_codebook(ply_path, layers_dict, device)
        return layers_dict

    def load_tiles(self, model: GaussianModel, ply_path: str, layers_dict: Dict[str, List[Layer]]) -> GaussianModel:
        tile_dir = os.path.splitext(ply_path)[0] + "_tiles"
        tiles = []
        i = 0
        while len(glob.glob(os.path.join(tile_dir, f"{i}.*"))) > 0:
            tile_path = os.path.join(tile_dir, f"{i}.ply")
            layers_dict = {k: [layer for layer in layers] for k, layers in layers_dict.items()}
            layers_dict = self.quantizer.load_enhencementlayers_codes(tile_path, layers_dict, model._xyz.device)
            layers_dict, xyz = self.quantizer.load_baselayer_codes(model.max_sh_degree, tile_path, layers_dict, model._xyz.device)
            tiles.append(Tile(layers_dict=layers_dict, gaussians=copy.deepcopy(model), xyz=xyz))
            i += 1
        return tiles

    def load_quantized_tiles(self, model: GaussianModel, ply_path: str) -> GaussianModel:
        layers_dict = self.load_codebooks(model.max_sh_degree, ply_path, model._xyz.device)
        tiles = self.load_tiles(model, ply_path, layers_dict)
        return self.dequantize_stitching(model, tiles)

    def pickup_quantized(self, max_sh_degree: int, input: str, output: str, layers: Dict[str, List[Layer]]):
        raise NotImplementedError("Pickup quantization is not implemented for TillingScalableQuantizer.")  # TODO
