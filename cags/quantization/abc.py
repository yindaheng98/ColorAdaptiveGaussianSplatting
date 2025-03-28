import abc
from typing import Dict, List, Tuple

import torch

from gaussian_splatting.gaussian_model import GaussianModel
from scalablevq import Layer


class InterfaceScalableQuantizer:
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

    # ---------------- load base layer ----------------

    @abc.abstractmethod
    def load_baselayer_codebook(self, max_sh_degree: int, ply_path: str, device) -> Dict[str, List[Layer]]:
        pass

    @abc.abstractmethod
    def load_baselayer_codes(self, layers_dict: Dict[str, List[Layer]], max_sh_degree: int, ply_path: str, device) -> Dict[str, List[Layer]]:
        pass

    def load_baselayer(self, max_sh_degree: int, ply_path: str, device) -> Dict[str, List[Layer]]:
        layers_dict = self.load_baselayer_codebook(max_sh_degree, ply_path, device)
        layers_dict = self.load_baselayer_codes(layers_dict, max_sh_degree, ply_path, device)
        return layers_dict

    # ---------------- load enhencement layer ----------------

    @abc.abstractmethod
    def load_enhencementlayer_codebooks(self, ply_path: str, layers_dict: Dict[str, List[Layer]], device) -> Dict[str, List[Layer]]:
        pass

    @abc.abstractmethod
    def load_enhencementlayer_codes(self, ply_path: str, layers_dict: Dict[str, List[Layer]], device) -> Dict[str, List[Layer]]:
        pass

    def load_enhencementlayers(self, ply_path: str, layers_dict: Dict[str, List[Layer]], device):
        layers_dict = self.load_enhencementlayer_codebooks(ply_path, layers_dict, device)
        layers_dict = self.load_enhencementlayer_codes(ply_path, layers_dict, device)
        return layers_dict
