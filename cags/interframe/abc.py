import abc
import copy
from typing import Tuple
import torch
import torch.nn as nn
from gaussian_splatting.gaussian_model import GaussianModel

from .rotation import quaternion_invert, quaternion_multiply, quaternion_to_matrix


class AbstractInterframeExtractor(abc.ABC):
    def init(self, frame: GaussianModel):
        self._last_frame = frame

    @abc.abstractmethod
    def diff_mask(self, frame: GaussianModel) -> torch.Tensor:
        pass

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
        diff_mask = self.diff_mask(frame)
        diff_frame = self.extract_by_mask(frame, diff_mask)
        self._last_frame = self.merge_next(diff_mask, diff_frame)
        return diff_frame, diff_mask

    def merge_next(self, diff_mask: torch.Tensor, diff_frame: GaussianModel) -> GaussianModel:
        assert self._last_frame is not None, ValueError("No initial frame provided. Call init() first.")
        self._last_frame = self.merge_by_mask(self._last_frame, diff_mask, diff_frame)
        return self._last_frame


class NoInterframeExtractor(AbstractInterframeExtractor):
    def diff_mask(self, frame: GaussianModel) -> torch.Tensor:
        torch.ones(frame.get_xyz.shape[0], dtype=torch.bool, device=frame.get_xyz.device)
