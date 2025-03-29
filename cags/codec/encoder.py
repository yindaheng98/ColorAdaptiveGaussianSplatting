from gaussian_splatting import GaussianModel
from cags.interframe import InterframeExtractor
from cags.tilequant import TillingScalableQuantizer


class Encoder:
    def __init__(
        self, frame_extractor: InterframeExtractor, frame_quantizer: TillingScalableQuantizer,
        tiling_first: bool = True, tiling_rest: bool = True
    ):
        self.frame_extractor = frame_extractor
        self.frame_quantizer = frame_quantizer
        self.tiling_first = tiling_first
        self.tiling_rest = tiling_rest

    def init(self, model: GaussianModel, ply_path: str):
        self.frame_extractor.init(model)
        if self.tiling_first:
            self.frame_quantizer.save_quantized_tiles(model, ply_path)
        else:
            self.frame_quantizer.quantizer.save_quantized(model, ply_path)

    def encode_next(self, model: GaussianModel, ply_path: str):
        diff_gaussians, diff_mask = self.frame_extractor.extract_next(model)
        if self.tiling_rest:
            layers_dict, tiles = self.frame_quantizer.quantize_tiling(diff_gaussians, update=False)
            self.frame_quantizer.save_tiles(ply_path, tiles)
        else:
            ids_dict, codebook_dict = self.frame_quantizer.quantizer.quantize(diff_gaussians, update_codebook=False)
            layers_dict = self.frame_quantizer.quantizer.layerize(diff_gaussians, ids_dict, codebook_dict, update_layers=False)
            self.frame_quantizer.quantizer.save_baselayer_codes(diff_gaussians, ply_path, layers_dict)
            self.frame_quantizer.quantizer.save_enhencementlayers_codes(ply_path, layers_dict)
