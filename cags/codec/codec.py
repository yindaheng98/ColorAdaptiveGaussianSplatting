import numpy as np
import torch
from gaussian_splatting import GaussianModel
from cags.tiling import AverageSplitTiling
from cags.interframe import InterframeExtractor
from cags.tilequant import TillingScalableQuantizer


class Codec:
    def __init__(
        self,
        frame_extractor: InterframeExtractor,
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
        np.savez_compressed(ply_path.replace(".ply", ".mask.npz"), mask=np.packbits(diff_mask.cpu().numpy(), axis=-1, bitorder='little'), n=diff_mask.shape[0])

    def decode_next(self, model: GaussianModel, ply_path: str) -> GaussianModel:
        if self.tiling_rest is not None:
            tiles = self.frame_quantizer.load_tiles(model, ply_path, self._layers_dict)
            diff_gaussians = self.frame_quantizer.dequantize_stitching(model, tiles)
        else:
            layers_dict, xyz = self.frame_quantizer.quantizer.load_baselayer_codes(model.max_sh_degree, ply_path, self._layers_dict, model._xyz.device)
            layers_dict = self.frame_quantizer.quantizer.load_enhencementlayers_codes(ply_path, layers_dict, model._xyz.device)
            ids_dict, codebook_dict = self.frame_quantizer.quantizer.delayerize(model.max_sh_degree, layers_dict)
            diff_gaussians = self.frame_quantizer.quantizer.dequantize(model, ids_dict, codebook_dict, xyz=xyz, replace=True)
        diff_mask_npz = np.load(ply_path.replace(".ply", ".mask.npz"))
        diff_mask = np.unpackbits(diff_mask_npz["mask"], count=diff_mask_npz["n"], axis=-1, bitorder='little')
        diff_mask = torch.from_numpy(diff_mask).to(device=model._xyz.device, dtype=torch.bool)
        return self.frame_extractor.merge_next(diff_mask, diff_gaussians)
