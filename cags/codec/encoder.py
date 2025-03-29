from gaussian_splatting import GaussianModel
from cags.interframe import InterframeExtractor
from cags.tilequant import TillingScalableQuantizer


class Encoder:
    def __init__(self, frame_extractor: InterframeExtractor, frame_quantizer: TillingScalableQuantizer):
        self.frame_extractor = frame_extractor
        self.frame_quantizer = frame_quantizer

    def init(self, model: GaussianModel, ply_path: str):
        self.frame_extractor.init(model)
        self.frame_quantizer.save_quantized_tiles(model, ply_path)

    def encode_next(self, model: GaussianModel, ply_path: str):
        diff_gaussians, diff_mask = self.frame_extractor.extract_next(model)
        self.frame_quantizer.save_quantized_tiles(diff_gaussians, ply_path)
