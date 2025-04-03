import os
import shutil
import numpy as np
import torch
from gaussian_splatting import GaussianModel
from cags.tiling import AverageSplitTiling
from cags.interframe import AbstractInterframeExtractor
from cags.tilequant import TillingScalableQuantizer


class Codec:
    def __init__(
        self,
        frame_extractor: AbstractInterframeExtractor,
        frame_quantizer: TillingScalableQuantizer,
        tiling_first: bool = True,
        tiling_rest: AverageSplitTiling = None,
    ):
        self.frame_extractor = frame_extractor
        self.frame_quantizer = frame_quantizer
        if tiling_rest is not None:
            assert tiling_first, "Tiling first must be set to True to use tiling rest"
        self.tiling_first = tiling_first
        self.tiling_rest = tiling_rest

        # encoding context
        self._tile_ids = None

        # decoding context
        self._layers_dict = None

        # pick up context
        self._ply_path_src = None

    def encode_init(self, model: GaussianModel, ply_path: str):
        if self.tiling_first:
            layers_dict, tiles, self._tile_ids = self.frame_quantizer.quantize_tiling(model, update=False)
            self.frame_quantizer.save_codebooks(ply_path, layers_dict)
            self.frame_quantizer.save_tiles(ply_path, tiles)
            model = self.frame_quantizer.tiling.sort_as_tiles(model, self._tile_ids)
        else:
            self.frame_quantizer.quantizer.save_quantized(model, ply_path)
        self.frame_extractor.init(model)

    def decode_init(self, model: GaussianModel, ply_path: str) -> GaussianModel:
        if self.tiling_first:
            self._layers_dict = self.frame_quantizer.load_codebooks(model.max_sh_degree, ply_path, model._xyz.device)
            tiles = self.frame_quantizer.load_tiles(model, ply_path,  self._layers_dict)
            model = self.frame_quantizer.dequantize_stitching(model, tiles)
        else:
            device = model._xyz.device
            self._layers_dict, xyz = self.frame_quantizer.quantizer.load_baselayer(model.max_sh_degree, ply_path, device)
            self._layers_dict = self.frame_quantizer.quantizer.load_enhencementlayers(ply_path, self._layers_dict, device)
            ids_dict, codebook_dict = self.frame_quantizer.quantizer.delayerize(model.max_sh_degree, self._layers_dict)
            model = self.frame_quantizer.quantizer.dequantize(model, ids_dict, codebook_dict, xyz=xyz, replace=True)
        self.frame_extractor.init(model)
        return model

    def pickup_init(self, max_sh_degree: int, ply_path_src: str, ply_path_dst: str, layer_dict: dict):
        if self.tiling_first:
            self.frame_quantizer.pickup_quantized(max_sh_degree, ply_path_src, ply_path_dst, layer_dict)
        else:
            self.frame_quantizer.quantizer.pickup_quantized(max_sh_degree, ply_path_src, ply_path_dst, layer_dict)
        self._ply_path_src = ply_path_src

    def save_mask(self, ply_path: str, diff_mask: torch.Tensor):
        np.savez_compressed(ply_path.replace(".ply", ".mask.npz"), mask=np.packbits(diff_mask.cpu().numpy(), axis=-1, bitorder='little'), n=diff_mask.shape[0])

    def encode_next(self, model: GaussianModel, ply_path: str):
        if self.tiling_first:
            model = self.frame_quantizer.tiling.sort_as_tiles(model, self._tile_ids)
        diff_gaussians, diff_mask = self.frame_extractor.extract_next(model)
        if self.tiling_rest is not None:
            tile_ids = self.tiling_rest.average_split(torch.arange(diff_gaussians._xyz.shape[0], device=diff_gaussians._xyz.device))
            layers_dict, tiles, _ = self.frame_quantizer.quantize_tiling(diff_gaussians, update=False, tile_gaussians_ids=tile_ids)
            self.frame_quantizer.save_tiles(ply_path, tiles)
        else:
            ids_dict, codebook_dict = self.frame_quantizer.quantizer.quantize(diff_gaussians, update_codebook=False)
            layers_dict = self.frame_quantizer.quantizer.layerize(diff_gaussians, ids_dict, codebook_dict, update_layers=False)
            self.frame_quantizer.quantizer.save_baselayer_codes(diff_gaussians, ply_path, layers_dict)
            self.frame_quantizer.quantizer.save_enhencementlayers_codes(ply_path, layers_dict)
        self.save_mask(ply_path, diff_mask)

    def load_mask(self, ply_path: str, device) -> torch.Tensor:
        diff_mask_npz = np.load(ply_path.replace(".ply", ".mask.npz"))
        diff_mask = np.unpackbits(diff_mask_npz["mask"], count=diff_mask_npz["n"], axis=-1, bitorder='little')
        diff_mask = torch.from_numpy(diff_mask).to(device=device, dtype=torch.bool)
        return diff_mask

    def decode_next(self, model: GaussianModel, ply_path: str) -> GaussianModel:
        if self.tiling_rest is not None:
            tiles = self.frame_quantizer.load_tiles(model, ply_path, self._layers_dict)
            diff_gaussians = self.frame_quantizer.dequantize_stitching(model, tiles)
        else:
            try:  # if `ply_path` is pick up frame, then the codebook can also be loaded from `ply_path`
                layers_dict, xyz = self.frame_quantizer.quantizer.load_baselayer(model.max_sh_degree, ply_path, model._xyz.device)
                layers_dict = self.frame_quantizer.quantizer.load_enhencementlayers(ply_path, layers_dict, model._xyz.device)
            except FileNotFoundError:
                layers_dict, xyz = self.frame_quantizer.quantizer.load_baselayer_codes(model.max_sh_degree, ply_path, self._layers_dict, model._xyz.device)
                layers_dict = self.frame_quantizer.quantizer.load_enhencementlayers_codes(ply_path, layers_dict, model._xyz.device)
            ids_dict, codebook_dict = self.frame_quantizer.quantizer.delayerize(model.max_sh_degree, layers_dict)
            diff_gaussians = self.frame_quantizer.quantizer.dequantize(model, ids_dict, codebook_dict, xyz=xyz, replace=True)
        diff_mask = self.load_mask(ply_path, model._xyz.device)
        return self.frame_extractor.merge_next(diff_mask, diff_gaussians)

    def pickup_next(self, max_sh_degree: int, ply_path_src: str, ply_path_dst: str, layer_dict: dict):
        if self.tiling_first:
            self.frame_quantizer.pickup_quantized_codebook(max_sh_degree, self._ply_path_src, ply_path_dst, layer_dict)
            self.frame_quantizer.pickup_quantized_codes(max_sh_degree, ply_path_src, ply_path_dst, layer_dict)
        else:
            self.frame_quantizer.quantizer.pickup_quantized_codebook(max_sh_degree, self._ply_path_src, ply_path_dst, layer_dict)
            self.frame_quantizer.quantizer.pickup_quantized_codes(max_sh_degree, ply_path_src, ply_path_dst, layer_dict)
            mask_src = ply_path_src.replace(".ply", ".mask.npz")
            mask_dst = ply_path_dst.replace(".ply", ".mask.npz")
            if mask_src == mask_dst:
                return
            if os.path.exists(mask_dst):
                os.remove(mask_dst)
            shutil.copy(mask_src, mask_dst)
