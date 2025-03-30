import abc
import os
import shutil
from typing import Dict, List, Tuple

import torch

from gaussian_splatting.gaussian_model import GaussianModel
from scalablevq import Layer
from reduced_3dgs.quantization import AbstractQuantizer


class InterfaceScalableQuantizer(AbstractQuantizer):
    @abc.abstractmethod
    def layerize(self, model: GaussianModel, ids_dict: Dict[str, torch.Tensor], codebook_dict: Dict[str, torch.Tensor], update_layers=False) -> Dict[str, List[Layer]]:
        pass

    @abc.abstractmethod
    def delayerize(self, max_sh_degree: int, layers_dict: Dict[str, List[Layer]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        pass

    # ---------------- save base layer ----------------

    @abc.abstractmethod
    def save_baselayer_codes(self, model: GaussianModel, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        pass

    @abc.abstractmethod
    def save_baselayer_codebook(self, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        pass

    def save_baselayer(self, model: GaussianModel, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        self.save_baselayer_codes(model, ply_path, layers_dict)
        self.save_baselayer_codebook(ply_path, layers_dict)

    # ---------------- save enhencement layers ----------------

    @abc.abstractmethod
    def save_enhencementlayers_codes(self, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        pass

    @abc.abstractmethod
    def save_enhencementlayers_codebook(self, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        pass

    def save_enhencementlayers(self, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        self.save_enhencementlayers_codes(ply_path, layers_dict)
        self.save_enhencementlayers_codebook(ply_path, layers_dict)

    # ---------------- save all quantized data ----------------

    def save_quantized(self, model: GaussianModel, ply_path: str):
        ids_dict, codebook_dict = self.quantize(model, update_codebook=False)
        layers_dict = self.layerize(model, ids_dict, codebook_dict, update_layers=False)
        self.save_baselayer(model, ply_path, layers_dict)
        self.save_enhencementlayers(ply_path, layers_dict)
        # ids_dict_orig = ids_dict
        # codebook_dict, ids_dict = self.delayerize(model, layers_dict)
        # for key in layers_dict.keys():
        #     if len(layers_dict[key]) <= 0:
        #         continue
        #     print(key, (self._codebook_dict[key][ids_dict_orig[key]] - codebook_dict[key][ids_dict[key]]).abs().max())

    # ---------------- load base layer ----------------

    @abc.abstractmethod
    def load_baselayer_codebook(self, max_sh_degree: int, ply_path: str, device) -> Dict[str, List[Layer]]:
        pass

    @abc.abstractmethod
    def load_baselayer_codes(self, max_sh_degree: int, ply_path: str, layers_dict: Dict[str, List[Layer]], device) -> Tuple[Dict[str, List[Layer]], torch.Tensor]:
        pass

    def load_baselayer(self, max_sh_degree: int, ply_path: str, device) -> Tuple[Dict[str, List[Layer]], torch.Tensor]:
        layers_dict = self.load_baselayer_codebook(max_sh_degree, ply_path, device)
        layers_dict, xyz = self.load_baselayer_codes(max_sh_degree, ply_path, layers_dict, device)
        return layers_dict, xyz

    # ---------------- load enhencement layer ----------------

    @abc.abstractmethod
    def load_enhencementlayers_codebook(self, ply_path: str, layers_dict: Dict[str, List[Layer]], device) -> Dict[str, List[Layer]]:
        pass

    @abc.abstractmethod
    def load_enhencementlayers_codes(self, ply_path: str, layers_dict: Dict[str, List[Layer]], device) -> Dict[str, List[Layer]]:
        pass

    def load_enhencementlayers(self, ply_path: str, layers_dict: Dict[str, List[Layer]], device):
        layers_dict = self.load_enhencementlayers_codebook(ply_path, layers_dict, device)
        layers_dict = self.load_enhencementlayers_codes(ply_path, layers_dict, device)
        return layers_dict

    # ---------------- load all quantized data ----------------

    def load_quantized(self, model: GaussianModel, ply_path: str) -> GaussianModel:
        device = model._xyz.device
        layers_dict, xyz = self.load_baselayer(model.max_sh_degree, ply_path, device)
        layers_dict = self.load_enhencementlayers(ply_path, layers_dict, device)
        ids_dict, codebook_dict = self.delayerize(model.max_sh_degree, layers_dict)
        return self.dequantize(model, ids_dict, codebook_dict, xyz=xyz, replace=True)

    # ---------------- pickup quantized data ----------------

    @abc.abstractmethod
    def filenames_quantized_codebook(self, max_sh_degree: int, ply_path: str, layer_dict: Dict[str, int]) -> List[str]:
        pass

    @abc.abstractmethod
    def filenames_quantized_codes(self, max_sh_degree: int, ply_path: str, layer_dict: Dict[str, int]) -> List[str]:
        pass

    def pickup_quantized_codebook(self, max_sh_degree: int, ply_path_src: str, ply_path_dst: str, layer_dict: Dict[str, int]):
        filenames_src = self.filenames_quantized_codebook(max_sh_degree, ply_path_src, layer_dict)
        filenames_dst = self.filenames_quantized_codebook(max_sh_degree, ply_path_dst, layer_dict)
        for src, dst in zip(filenames_src, filenames_dst):
            if src == dst:
                continue
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy(src, dst)

    def pickup_quantized_codes(self, max_sh_degree: int, ply_path_src: str, ply_path_dst: str, layer_dict: Dict[str, int]):
        filenames_src = self.filenames_quantized_codes(max_sh_degree, ply_path_src, layer_dict)
        filenames_dst = self.filenames_quantized_codes(max_sh_degree, ply_path_dst, layer_dict)
        for src, dst in zip(filenames_src, filenames_dst):
            if src == dst:
                continue
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy(src, dst)

    def pickup_quantized(self, max_sh_degree: int, ply_path_src: str, ply_path_dst: str, layer_dict: Dict[str, int]):
        self.pickup_quantized_codebook(max_sh_degree, ply_path_src, ply_path_dst, layer_dict)
        self.pickup_quantized_codes(max_sh_degree, ply_path_src, ply_path_dst, layer_dict)
