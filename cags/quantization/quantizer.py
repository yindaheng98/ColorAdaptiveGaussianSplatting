import math
from typing import List, Dict
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from gaussian_splatting import GaussianModel
from reduced_3dgs.quantization import ExcludeZeroSHQuantizer
from scalablevq import encode_layers, Layer


def expand_base_layer(layer: Layer, zero_mask: torch.Tensor):
    codes = torch.zeros(zero_mask.shape[0], dtype=layer.codes.dtype, device=layer.codes.device)
    codes[~zero_mask] = layer.codes + (1 << layer.n_bit)
    codebook = torch.cat([torch.zeros(1, dtype=layer.codebook.dtype, device=layer.codebook.device), layer.codebook])
    cluster_centers = torch.cat([torch.zeros((1, layer.cluster_centers.shape[1]), dtype=layer.cluster_centers.dtype, device=layer.cluster_centers.device), layer.cluster_centers])
    return Layer(codes=codes, codebook=codebook, cluster_centers=cluster_centers, n_bit=layer.n_bit + 1, n_leaf=layer.n_leaf + 1)


class ScalableQuantizer(ExcludeZeroSHQuantizer):
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
        return encode_layers(values, ids, codebook, n_bits_proposal)

    def encode_layers_exclude_zero(self, values: torch.Tensor, ids: torch.Tensor, codebook: torch.Tensor, n_bits_proposal: List[int]):
        if codebook.shape[0] <= 1:  # all zero from ExcludeZeroQuantizer.generate_codebook
            return []
        zeros_mask = ids == 0
        nonzero_values, nonzero_ids = values[~zeros_mask], ids[~zeros_mask]
        nonzero_codebook = codebook[nonzero_ids.unique()]
        layers = encode_layers(nonzero_values, nonzero_ids, nonzero_codebook, self.n_bits_proposal_features_dc)
        return [expand_base_layer(layers[0], zeros_mask)] + layers[1:]

    def produce_layers_features_dc(self, ids, codebook):
        return self.encode_layers(self.model._features_dc.detach().squeeze(1), ids, codebook, self.n_bits_proposal_features_dc)

    def produce_layers_features_rest(self, sh_degree, ids, codebook):
        sh_idx_start, sh_idx_end = (sh_degree + 1) ** 2 - 1, (sh_degree + 2) ** 2 - 1
        features_rest_flatten = self.model._features_rest.detach().transpose(1, 2).flatten(0, 1)
        features_rest = features_rest_flatten[:, sh_idx_start:sh_idx_end]
        ids_reshaped = ids.reshape(-1)
        layers = self.encode_layers_exclude_zero(features_rest, ids_reshaped, codebook, self.n_bits_proposal_features_rest[sh_degree])
        return layers

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
        self._codebook_dict = codebook_dict
        layers_dict = self.produce_layers(codebook_dict, ids_dict)
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('rot_re', self.force_code_dtype or f"u{math.ceil(layers_dict['rotation_re'][0].n_bit / 8)}"),
            ('rot_im', self.force_code_dtype or f"u{math.ceil(layers_dict['rotation_im'][0].n_bit / 8)}"),
            ('opacity', self.force_code_dtype or f"u{math.ceil(layers_dict['opacity'][0].n_bit / 8)}"),
            ('scale', self.force_code_dtype or f"u{math.ceil(layers_dict['scaling'][0].n_bit / 8)}"),
            ('f_dc', self.force_code_dtype or f"u{math.ceil(layers_dict['features_dc'][0].n_bit / 8)}"),
        ]
        for sh_degree in range(model.max_sh_degree):
            if not layers_dict[f"features_rest_{sh_degree}"]:
                continue
            force_code_dtype = self.force_code_dtype or f"u{math.ceil(layers_dict[f'features_rest_{sh_degree}'][0].n_bit / 8)}"
            dtype_full.extend([
                (f'f_rest_{sh_degree}_0', force_code_dtype),
                (f'f_rest_{sh_degree}_1', force_code_dtype),
                (f'f_rest_{sh_degree}_2', force_code_dtype),
            ])
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
            if not layers_dict[f"features_rest_{sh_degree}"]:
                continue
            features_rest = layers_dict[f'features_rest_{sh_degree}'][0].codes.reshape(-1, 3).cpu().numpy()
            data_full.extend(np.array_split(features_rest, 3, axis=1))

        elements = np.rec.fromarrays([data.squeeze(-1) for data in data_full], dtype=dtype_full)
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(ply_path)
