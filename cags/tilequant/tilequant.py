import glob
import os
from typing import Dict, List, NamedTuple, Tuple
import tqdm
from gaussian_splatting.gaussian_model import GaussianModel
from cags.quantization import InterfaceScalableQuantizer
from cags.tiling import AbstractTiling
from scalablevq import Layer


class Tile(NamedTuple):
    layers_dict: Dict[str, List[Layer]]
    gaussians: GaussianModel


class TillingScalableQuantizer:

    def __init__(self, quantizer: InterfaceScalableQuantizer, tiling: AbstractTiling):
        self.quantizer = quantizer
        self.tiling = tiling

    def quantize_tiling(self, model: GaussianModel) -> Tuple[Dict[str, List[Layer]], List[Tile]]:
        ids_dict, codebook_dict = self.quantizer.quantize(model, update_codebook=True)
        layers_dict = self.quantizer.layerize(model, ids_dict, codebook_dict, update_layers=True)
        tiles = []
        for tile in tqdm.tqdm(self.tiling.tiling(model), desc="Quantizing tiles"):
            ids_dict, codebook_dict = self.quantizer.quantize(tile, update_codebook=False)
            layers_dict = self.quantizer.layerize(tile, ids_dict, codebook_dict, update_layers=False)
            tiles.append(Tile(gaussians=tile, layers_dict=layers_dict))
        return layers_dict, tiles

    def dequantize_stitching(self, model: GaussianModel, tiles: List[Tile]) -> GaussianModel:
        raise NotImplementedError

    def save_quantized_tiles(self, model: GaussianModel, ply_path: str):
        layers_dict, tiles = self.quantize_tiling(model)
        self.quantizer.save_baselayer_codebook(ply_path, layers_dict)
        self.quantizer.save_enhencementlayers_codebook(ply_path, layers_dict)
        tile_dir = os.path.splitext(ply_path)[0] + "_tiles"
        os.makedirs(tile_dir, exist_ok=True)
        for i, tile in enumerate(tqdm.tqdm(tiles, desc="Saving tiles")):
            tile_path = os.path.join(tile_dir, f"{i}.ply")
            self.quantizer.save_baselayer_codes(tile.gaussians, tile_path, tile.layers_dict)
            self.quantizer.save_enhencementlayers_codes(tile_path, tile.layers_dict)

    def load_quantized_tiles(self, model: GaussianModel, ply_path: str) -> GaussianModel:
        layers_dict = self.quantizer.load_baselayer_codebook(model.max_sh_degree, ply_path, model._xyz.device)
        layers_dict = self.quantizer.load_enhencementlayers_codebook(ply_path, layers_dict, model._xyz.device)
        tile_dir = os.path.splitext(ply_path)[0] + "_tiles"
        tiles = []
        i = 0
        while len(glob.glob(os.path.join(tile_dir, f"{i}.*"))) > 0:
            tile_path = os.path.join(tile_dir, f"{i}.ply")
            layers_dict = {k: [layer for layer in layers] for k, layers in layers_dict.items()}
            layers_dict = self.quantizer.load_enhencementlayers_codes(tile_path, layers_dict, model._xyz.device)
            layers_dict, xyz = self.quantizer.load_baselayer_codes(model.max_sh_degree, tile_path, layers_dict, model._xyz.device)
            tiles.append(Tile(layers_dict=layers_dict, gaussians=None))
            i += 1
        return self.dequantize_stitching(model, tiles)
