import abc
from typing import Tuple
import torch
import torch.nn as nn
from gaussian_splatting.gaussian_model import GaussianModel

from .rotation import quaternion_invert, quaternion_multiply, quaternion_to_matrix


class AbstractInterframeExtractor(abc.ABC):
    @abc.abstractmethod
    def init(self, frame: GaussianModel):
        self._last_frame = frame

    @abc.abstractmethod
    def extract_next(self, frame: GaussianModel) -> Tuple[GaussianModel, torch.Tensor]:
        pass

    @abc.abstractmethod
    def merge_next(self, diff_mask: torch.Tensor, diff_frame: GaussianModel) -> GaussianModel:
        pass


class NoInterframeExtractor(AbstractInterframeExtractor):
    def init(self, frame: GaussianModel):
        pass

    def extract_next(self, frame: GaussianModel) -> Tuple[GaussianModel, torch.Tensor]:
        return frame, torch.ones(frame.get_xyz.shape[0], dtype=torch.bool, device=frame.get_xyz.device)

    def merge_next(self, diff_mask: torch.Tensor, diff_frame: GaussianModel) -> GaussianModel:
        return diff_frame
