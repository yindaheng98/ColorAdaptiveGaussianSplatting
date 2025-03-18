import math
from typing import List, Dict
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from gaussian_splatting import GaussianModel
from reduced_3dgs.quantization import ExcludeZeroQuantizer
from scalablevq import encode_layers, Layer


def array2record(array: torch.Tensor, perfix, n_cols, dtype):
    dtype_full = [(f'{perfix}_{i}', dtype) for i in range(n_cols)] if n_cols > 1 else [(perfix, dtype)]
    data_full = map(lambda x: x.squeeze(-1), np.array_split(array.cpu().numpy(), n_cols, axis=1))
    record = np.rec.fromarrays(data_full, dtype=dtype_full)
    return record


class ScalableQuantizer(ExcludeZeroQuantizer):
    def __init__(
        self, model: GaussianModel,
        n_bits_proposal: List[int] = [4, 2, 2, 2, 2],
        n_bits_proposal_rotation_re: List[int] = None,
        n_bits_proposal_rotation_im: List[int] = None,
        n_bits_proposal_opacity: List[int] = None,
        n_bits_proposal_scaling: List[int] = None,
        n_bits_proposal_features_dc: List[int] = None,
        n_bits_proposal_features_rest: List[List[int]] = [],
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.n_bits_proposal_rotation_re = n_bits_proposal_rotation_re or n_bits_proposal.copy()
        self.n_bits_proposal_rotation_im = n_bits_proposal_rotation_im or n_bits_proposal.copy()
        self.n_bits_proposal_opacity = n_bits_proposal_opacity or n_bits_proposal.copy()
        self.n_bits_proposal_scaling = n_bits_proposal_scaling or n_bits_proposal.copy()
        self.n_bits_proposal_features_dc = n_bits_proposal_features_dc or n_bits_proposal.copy()
        self.n_bits_proposal_features_rest = [(n_bits_proposal_features_rest[i] if len(n_bits_proposal_features_rest) > i else n_bits_proposal.copy()) for i in range(model.max_sh_degree)]

    def encode_layers(self, values: torch.Tensor, ids: torch.Tensor, codebook: torch.Tensor, n_bits_proposal: List[int]):
        zeros_mask = (values.abs() < self.treat_as_zero).all(-1)
        if zeros_mask.all():
            return []
        if zeros_mask.sum() <= self.extract_zero_thr * values.shape[0]:
            return encode_layers(values, ids, codebook, n_bits_proposal)
        nonzero_values, nonzero_ids = values[~zeros_mask], ids[~zeros_mask]
        nonzero_codebook = codebook[nonzero_ids.unique()]
        layers = encode_layers(nonzero_values, nonzero_ids, nonzero_codebook, self.n_bits_proposal_features_dc)
        return layers

    def produce_layers_features_dc(self, ids, codebook):
        return self.encode_layers(self.model._features_dc.detach().squeeze(1), ids, codebook, self.n_bits_proposal_features_dc)

    def produce_layers_features_rest(self, sh_degree, ids, codebook):
        sh_idx_start, sh_idx_end = (sh_degree + 1) ** 2 - 1, (sh_degree + 2) ** 2 - 1
        features_rest_flatten = self.model._features_rest.detach().transpose(1, 2).flatten(0, 1)
        features_rest = features_rest_flatten[:, sh_idx_start:sh_idx_end]
        layers = self.encode_layers(features_rest, ids.reshape(-1), codebook, self.n_bits_proposal_features_rest[sh_degree])
        return layers  # TODO: reshape layers

    def produce_layers_rotation_re(self, ids, codebook):
        return self.encode_layers(self.model.get_rotation.detach()[:, 0:1], ids, codebook, self.n_bits_proposal_rotation_re)

    def produce_layers_rotation_im(self, ids, codebook):
        return self.encode_layers(self.model.get_rotation.detach()[:, 1:], ids, codebook, self.n_bits_proposal_rotation_im)

    def produce_layers_opacity(self, ids, codebook):
        return self.encode_layers(self.model._opacity.detach(), ids, codebook, self.n_bits_proposal_opacity)

    def produce_layers_scaling(self, ids, codebook):
        return self.encode_layers(self.model._scaling.detach(), ids, codebook, self.n_bits_proposal_scaling)

    def produce_layers(self, codebook_dict: Dict[str, torch.Tensor], ids_dict: Dict[str, torch.Tensor]):
        layers_dict: Dict[str, List[Layer]] = {}

        layers_dict["features_dc"] = self.produce_layers_features_dc(ids_dict["features_dc"].squeeze(1), codebook_dict["features_dc"])
        for sh_degree in range(self.model.max_sh_degree):
            layers_dict[f'features_rest_{sh_degree}'] = self.produce_layers_features_rest(sh_degree, ids_dict[f'features_rest_{sh_degree}'], codebook_dict[f'features_rest_{sh_degree}'])

        layers_dict["rotation_re"] = self.produce_layers_rotation_re(ids_dict["rotation_re"], codebook_dict["rotation_re"])
        layers_dict["rotation_im"] = self.produce_layers_rotation_im(ids_dict["rotation_im"], codebook_dict["rotation_im"])

        layers_dict["opacity"] = self.produce_layers_opacity(ids_dict["opacity"], codebook_dict["opacity"])
        layers_dict["scaling"] = self.produce_layers_scaling(ids_dict["scaling"], codebook_dict["scaling"])
        return layers_dict

    def save_quantized(self, ply_path: str):
        model = self.model
        codebook_dict, ids_dict = self.produce_clusters(self._codebook_dict)
        layers_dict = self.produce_layers(codebook_dict, ids_dict)
        pass  # TODO
