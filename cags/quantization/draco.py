import math
import os
import platform
import subprocess
from typing import List, Dict
import numpy as np
from plyfile import PlyData, PlyElement
import torch
from gaussian_splatting.gaussian_model import GaussianModel
from scalablevq import Layer
from .quantizer import ScalableQuantizer


class DracoCompressedScalableQuantizer(ScalableQuantizer):
    def __init__(
        self,
        draco_encoder_executable: str = "./build/Release/draco_encoder.exe" if platform.system() == "Windows" else "./build/draco_encoder",
        draco_decoder_executable: str = "./build/Release/draco_decoder.exe" if platform.system() == "Windows" else "./build/draco_decoder",
        draco_qp=16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.draco_encoder_executable = draco_encoder_executable
        self.draco_decoder_executable = draco_decoder_executable
        self.draco_qp = draco_qp
        self.draco_q = {}

    def baselayer_ply_dtype(self, max_sh_degree: int, layers_dict: Dict[str, List[Layer]]):
        force_n_bit_rotation = max(layers_dict['rotation_re'][0].n_bit, layers_dict['rotation_im'][0].n_bit)
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('rot_re', self.force_code_dtype or f"u{math.ceil(force_n_bit_rotation / 8)}"),
            ('rot_im', self.force_code_dtype or f"u{math.ceil(force_n_bit_rotation / 8)}"),
            ('opacity', self.force_code_dtype or f"u{math.ceil(layers_dict['opacity'][0].n_bit / 8)}"),
            ('scale', self.force_code_dtype or f"u{math.ceil(layers_dict['scaling'][0].n_bit / 8)}"),
            ('f_dc', self.force_code_dtype or f"u{math.ceil(layers_dict['features_dc'][0].n_bit / 8)}"),
        ]
        force_n_bit_features_rest = 1
        for sh_degree in range(max_sh_degree):
            if len(layers_dict[f"features_rest_{sh_degree}"]) <= 0:
                continue
            force_n_bit_features_rest = max(force_n_bit_features_rest, layers_dict[f'features_rest_{sh_degree}'][0].n_bit)
        for sh_degree in range(max_sh_degree):
            force_code_dtype = self.force_code_dtype or f"u{math.ceil(force_n_bit_features_rest / 8)}"
            dtype_full.extend([
                (f'f_rest_{sh_degree}_0', force_code_dtype),
                (f'f_rest_{sh_degree}_1', force_code_dtype),
                (f'f_rest_{sh_degree}_2', force_code_dtype),
            ])
        self.draco_q['rotation'] = force_n_bit_rotation
        self.draco_q['opacity'] = layers_dict['opacity'][0].n_bit
        self.draco_q['scaling'] = layers_dict['scaling'][0].n_bit
        self.draco_q['features_dc'] = layers_dict['features_dc'][0].n_bit
        self.draco_q['features_rest'] = force_n_bit_features_rest
        return dtype_full

    def baselayer_ply_data(self, model: GaussianModel, layers_dict: Dict[str, List[Layer]]):
        data_full = [
            *np.array_split(model._xyz.detach().cpu().numpy(), 3, axis=1),
            *np.array_split(torch.zeros_like(model._xyz).detach().cpu().numpy(), 3, axis=1),
            layers_dict["rotation_re"][0].codes.unsqueeze(-1).cpu().numpy(),
            layers_dict["rotation_im"][0].codes.unsqueeze(-1).cpu().numpy(),
            layers_dict["opacity"][0].codes.unsqueeze(-1).cpu().numpy(),
            layers_dict["scaling"][0].codes.unsqueeze(-1).cpu().numpy(),
            layers_dict["features_dc"][0].codes.unsqueeze(-1).cpu().numpy(),
        ]
        for sh_degree in range(model.max_sh_degree):
            if len(layers_dict[f"features_rest_{sh_degree}"]) <= 0:
                features_rest = torch.zeros((model._xyz.shape[0], 3), dtype=torch.int).detach().cpu().numpy()
            else:
                features_rest = layers_dict[f'features_rest_{sh_degree}'][0].codes.reshape(-1, 3).cpu().numpy()
            data_full.extend(np.array_split(features_rest, 3, axis=1))
        return data_full

    def save_baselayer_codes(self, model: GaussianModel, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        dtype_full = self.baselayer_ply_dtype(model.max_sh_degree, layers_dict)
        data_full = self.baselayer_ply_data(model, layers_dict)

        elements = np.rec.fromarrays([data.squeeze(-1) for data in data_full], dtype=dtype_full)
        el = PlyElement.describe(elements, 'vertex')

        src_path = os.path.splitext(ply_path)[0] + ".drcsource.ply"
        PlyData([el]).write(src_path)

        drc_path = os.path.splitext(ply_path)[0] + ".drc"
        subprocess.check_call([
            self.draco_encoder_executable,
            "-i", src_path, "-o", drc_path, "-cl", str(0),
            "-qp", str(self.draco_qp),
            "-qscale", str(self.draco_q['scaling']),
            "-qrotation", str(self.draco_q['rotation']),
            "-qopacity", str(self.draco_q['opacity']),
            "-qfeaturedc", str(self.draco_q['features_dc']),
            "-qfeaturerest", str(self.draco_q['features_rest']),
        ])

    def load_baselayer_codes(self, max_sh_degree: int, ply_path: str, layers_dict: Dict[str, List[Layer]], device):
        drc_path = os.path.splitext(ply_path)[0] + ".drc"
        extract_path = os.path.splitext(ply_path)[0] + ".drcdecode.ply"
        subprocess.check_call([
            self.draco_decoder_executable,
            "-i", drc_path, "-o", extract_path,
        ])
        true_max_sh_degree = 0
        for sh_degree in range(max_sh_degree):
            if len(layers_dict[f"features_rest_{sh_degree}"]) <= 0:
                break
            true_max_sh_degree += 1
        return super().load_baselayer_codes(true_max_sh_degree, extract_path, layers_dict, device)
