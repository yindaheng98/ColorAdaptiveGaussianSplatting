import copy
from typing import Tuple
import torch
import torch.nn as nn
from gaussian_splatting.gaussian_model import GaussianModel


class InterframeExtractor:
    def __init__(
        self,
        diff_thr_xyz: float = 0.1,
        diff_thr_rotation: float = 0.1,
        diff_thr_opacity: float = 0.1,
        diff_thr_scaling: float = 0.1,
        diff_thr_feature_dc: float = 0.1,
        diff_thr_feature_rest: float = 0.1,
    ):
        self.diff_thr_xyz = diff_thr_xyz
        self.diff_thr_rotation = diff_thr_rotation
        self.diff_thr_opacity = diff_thr_opacity
        self.diff_thr_scaling = diff_thr_scaling
        self.diff_thr_feature_dc = diff_thr_feature_dc
        self.diff_thr_feature_rests = diff_thr_feature_rest

        self._last_frame = None

    def init(self, frame: GaussianModel):
        self._last_frame = frame

    def diff_mask(self, frame: GaussianModel, last_frame: GaussianModel) -> GaussianModel:
        def diff_mask_attr(attr: torch.Tensor, last_attr: torch.Tensor, diff_thr: float) -> GaussianModel:
            return (attr - last_attr).abs().flatten(1).max(dim=1).values > diff_thr
        with torch.no_grad():
            diff_mask = diff_mask_attr(frame.get_xyz, last_frame.get_xyz, self.diff_thr_xyz)
            diff_mask |= diff_mask_attr(frame.get_rotation, last_frame.get_rotation, self.diff_thr_rotation)
            diff_mask |= diff_mask_attr(frame.get_opacity, last_frame.get_opacity, self.diff_thr_opacity)
            diff_mask |= diff_mask_attr(frame.get_scaling, last_frame.get_scaling, self.diff_thr_scaling)
            diff_mask |= diff_mask_attr(frame.get_features_dc, last_frame.get_features_dc, self.diff_thr_feature_dc)
            diff_mask |= diff_mask_attr(frame.get_features_rest, last_frame.get_features_rest, self.diff_thr_feature_rests)
        return diff_mask

    def extract_by_mask(self, frame: GaussianModel, diff_mask: torch.Tensor) -> GaussianModel:
        diff_frame = copy.deepcopy(frame)
        with torch.no_grad():
            diff_frame._xyz = nn.Parameter(diff_frame._xyz[diff_mask, ...])
            diff_frame._rotation = nn.Parameter(diff_frame._rotation[diff_mask, ...])
            diff_frame._opacity = nn.Parameter(diff_frame._opacity[diff_mask, ...])
            diff_frame._scaling = nn.Parameter(diff_frame._scaling[diff_mask, ...])
            diff_frame._features_dc = nn.Parameter(diff_frame._features_dc[diff_mask, ...])
            diff_frame._features_rest = nn.Parameter(diff_frame._features_rest[diff_mask, ...])
        return diff_frame

    def merge_by_mask(self, frame: GaussianModel, diff_mask: torch.Tensor, diff_frame: GaussianModel) -> GaussianModel:
        with torch.no_grad():
            frame._xyz[diff_mask, ...] = diff_frame._xyz
            frame._rotation[diff_mask, ...] = diff_frame._rotation
            frame._opacity[diff_mask, ...] = diff_frame._opacity
            frame._scaling[diff_mask, ...] = diff_frame._scaling
            frame._features_dc[diff_mask, ...] = diff_frame._features_dc
            frame._features_rest[diff_mask, ...] = diff_frame._features_rest
        return frame

    def extract_next(self, frame: GaussianModel) -> Tuple[GaussianModel, torch.Tensor]:
        assert self._last_frame is not None, ValueError("No initial frame provided. Call init() first.")
        diff_mask = self.diff_mask(frame, self._last_frame)
        diff_frame = self.extract_by_mask(frame, diff_mask)
        self._last_frame = self.merge_by_mask(frame, diff_mask, diff_frame)
        return diff_frame, diff_mask
