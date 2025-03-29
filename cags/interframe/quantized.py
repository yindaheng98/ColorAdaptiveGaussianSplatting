import torch
from gaussian_splatting.gaussian_model import GaussianModel
from reduced_3dgs.quantization import AbstractQuantizer

from .abc import AbstractInterframeExtractor


class QuantizedInterframeExtractor(AbstractInterframeExtractor):
    def __init__(self, quantizer: AbstractQuantizer, diff_thr_xyz_stdfactor: float = 0.08):
        self.quantizer = quantizer
        self.diff_thr_xyz = diff_thr_xyz_stdfactor

    def init(self, frame: GaussianModel):
        self._last_frame = frame
        self._ids_dict, _ = self.quantizer.quantize(frame, update_codebook=False)

    def diff_mask_xyz(self, frame: GaussianModel, last_frame: GaussianModel) -> GaussianModel:
        diff = last_frame.get_xyz - frame.get_xyz
        std = torch.cat([frame.get_xyz, last_frame.get_xyz], dim=0).std(dim=0).min()
        return (diff.abs() > std * self.diff_thr_xyz).any(dim=1)

    def diff_mask(self, frame: GaussianModel) -> GaussianModel:
        last_frame = self._last_frame
        last_ids_dict = self._ids_dict
        ids_dict, _ = self.quantizer.quantize(frame, update_codebook=False)
        with torch.no_grad():
            diff_mask = self.diff_mask_xyz(frame, last_frame)
            diff_mask |= ids_dict["rotation_re"] != last_ids_dict["rotation_re"]
            diff_mask |= ids_dict["rotation_im"] != last_ids_dict["rotation_im"]
            diff_mask |= ids_dict["opacity"] != last_ids_dict["opacity"]
            diff_mask |= ids_dict["scaling"] != last_ids_dict["scaling"]
            diff_mask |= (ids_dict["features_dc"] != last_ids_dict["features_dc"]).any(-1)
            for sh_degree in range(frame.max_sh_degree):
                if f"features_rest_{sh_degree}" not in ids_dict:
                    continue
                if f"features_rest_{sh_degree}" not in last_ids_dict:
                    continue
                diff_mask |= (ids_dict[f"features_rest_{sh_degree}"] != last_ids_dict[f"features_rest_{sh_degree}"]).any(-1)
        return diff_mask
