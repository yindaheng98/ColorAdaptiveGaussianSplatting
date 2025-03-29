import copy
from typing import Tuple
import torch
import torch.nn as nn
from gaussian_splatting.gaussian_model import GaussianModel

from .abc import AbstractInterframeExtractor
from .rotation import quaternion_invert, quaternion_multiply, quaternion_to_matrix


class InterframeExtractor(AbstractInterframeExtractor):
    def __init__(
        self,
        diff_thr_xyz_stdfactor: float = 0.08,
        diff_thr_rotation_eular_degree: float = 30,
        diff_thr_opacity_absolute: float = 0.2,
        diff_thr_scaling_stdfactor: float = 0.8,
        diff_thr_feature_dc_stdfactor: float = 0.8,
        diff_thr_feature_rest_stdfactor: float = 1.0,
    ):
        """
        Args:
            diff_thr_xyz_std_factor (float): Threshold for xyz difference based on standard deviation. idea from 3-sigma rule https://en.wikipedia.org/wiki/Three-sigma_rule
            diff_thr_rotation_eular_degree (float): Threshold for rotation difference in Euler angles (degree).
            diff_thr_opacity_absolute (float): Threshold for absolute opacity difference.
            diff_thr_scaling_stdfactor (float): Threshold for scaling difference based on standard deviation.
            diff_thr_feature_dc (float): Threshold for feature difference in DC component.
            diff_thr_feature_rest (float): Threshold for feature difference in rest components.
        """
        self.diff_thr_xyz = diff_thr_xyz_stdfactor
        self.diff_thr_rotation = diff_thr_rotation_eular_degree
        self.diff_thr_opacity = diff_thr_opacity_absolute
        self.diff_thr_scaling = diff_thr_scaling_stdfactor
        self.diff_thr_feature_dc = diff_thr_feature_dc_stdfactor
        self.diff_thr_feature_rests = diff_thr_feature_rest_stdfactor

        self._last_frame = None

    def init(self, frame: GaussianModel):
        self._last_frame = frame

    def diff_mask_xyz(self, frame: GaussianModel, last_frame: GaussianModel) -> GaussianModel:
        diff = last_frame.get_xyz - frame.get_xyz
        std = torch.cat([frame.get_xyz, last_frame.get_xyz], dim=0).std(dim=0).min()
        return (diff.abs() > std * self.diff_thr_xyz).any(dim=1)

    def diff_mask_rotation(self, frame: GaussianModel, last_frame: GaussianModel) -> GaussianModel:
        diff_quaternion = quaternion_multiply(quaternion_invert(last_frame.get_rotation), frame.get_rotation)
        diff_matrix = quaternion_to_matrix(diff_quaternion)
        diff_euler = torch.acos((torch.vmap(torch.trace)(diff_matrix) - 1) / 2) * 180 / torch.pi
        diff_degree = torch.min(180 - diff_euler, diff_euler)
        return diff_degree > self.diff_thr_rotation

    def diff_mask_opacity(self, frame: GaussianModel, last_frame: GaussianModel) -> GaussianModel:
        diff = last_frame.get_opacity - frame.get_opacity
        return (diff.abs() > self.diff_thr_opacity).any(dim=1)

    def diff_mask_scaling(self, frame: GaussianModel, last_frame: GaussianModel) -> GaussianModel:
        diff = last_frame.get_scaling - frame.get_scaling
        std = torch.cat([frame.get_scaling, last_frame.get_scaling], dim=0).std(dim=0)
        return (diff.abs() > std * self.diff_thr_scaling).any(dim=1)

    def diff_mask(self, frame: GaussianModel, last_frame: GaussianModel) -> GaussianModel:
        def diff_mask_attr(attr: torch.Tensor, last_attr: torch.Tensor, diff_thr: float) -> GaussianModel:
            flatten_attr, flatten_last_attr = attr.flatten(1), last_attr.flatten(1)
            std = torch.cat([flatten_attr, flatten_last_attr], dim=0).std(dim=0)
            return ((flatten_attr - flatten_last_attr).abs() > std * diff_thr).any(dim=1)
        with torch.no_grad():
            diff_mask = self.diff_mask_xyz(frame, last_frame)
            diff_mask |= self.diff_mask_rotation(frame, last_frame)  # most difference comes from here
            diff_mask |= self.diff_mask_opacity(frame, last_frame)
            diff_mask |= self.diff_mask_scaling(frame, last_frame)
            diff_mask |= diff_mask_attr(frame.get_features_dc, last_frame.get_features_dc, self.diff_thr_feature_dc)
            diff_mask |= diff_mask_attr(frame.get_features_rest[:, :3, ...], last_frame.get_features_rest[:, :3, ...], self.diff_thr_feature_rests)
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
        self._last_frame = self.merge_next(diff_mask, diff_frame)
        return diff_frame, diff_mask

    def merge_next(self, diff_mask: torch.Tensor, diff_frame: GaussianModel) -> GaussianModel:
        assert self._last_frame is not None, ValueError("No initial frame provided. Call init() first.")
        self._last_frame = self.merge_by_mask(self._last_frame, diff_mask, diff_frame)
        return self._last_frame
