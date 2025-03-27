import os
import math
from typing import Callable, List, Dict
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from gaussian_splatting import GaussianModel
from reduced_3dgs.quantization import ExcludeZeroSHQuantizer
from scalablevq import encode_layers, extract_layers, Layer


def expand_base_layer(layer: Layer, zero_mask: torch.Tensor):
    codes = torch.zeros(zero_mask.shape[0], dtype=layer.codes.dtype, device=layer.codes.device)
    codes[~zero_mask] = layer.codes + (1 << layer.n_bit)
    codebook = torch.cat([torch.zeros(1, dtype=layer.codebook.dtype, device=layer.codebook.device), layer.codebook])
    cluster_centers = torch.cat([torch.zeros((1, layer.cluster_centers.shape[1]), dtype=layer.cluster_centers.dtype, device=layer.cluster_centers.device), layer.cluster_centers])
    return layer._replace(codes=codes, codebook=codebook, cluster_centers=cluster_centers)


def shrink_base_layer(layer: Layer):
    zero_mask = layer.codes == 0
    codes = layer.codes[~zero_mask] - (1 << layer.n_bit)
    codebook = layer.codebook[1:, ...]
    cluster_centers = layer.cluster_centers[1:, ...]
    return layer._replace(codes=codes, codebook=codebook, cluster_centers=cluster_centers), zero_mask


def load_layer(path: str, device: torch.device):
    layer = np.load(path)
    return Layer(
        codes=torch.tensor(layer["codes"], device=device),
        codebook=torch.tensor(layer["codebook"], device=device),
        cluster_centers=torch.tensor(layer["cluster_centers"], device=device),
        n_bit=layer["n_bit"].item(),
        n_leaf=layer["n_leaf"].item(),
    )


class ScalableQuantizer(ExcludeZeroSHQuantizer):
    def __init__(
        self,
        max_sh_degree=3,
        n_bit_baselayer: int = 4,
        n_bits_proposal: int | List[int] | Callable[[int, torch.Tensor, torch.Tensor], List[int]] = [2, 2, 2, 2],
        n_bit_baselayer_rotation_re: int = None,
        n_bits_proposal_rotation_re: int | List[int] | Callable[[int, torch.Tensor, torch.Tensor], List[int]] = None,
        n_bit_baselayer_rotation_im: int = None,
        n_bits_proposal_rotation_im: int | List[int] | Callable[[int, torch.Tensor, torch.Tensor], List[int]] = None,
        n_bit_baselayer_opacity: int = None,
        n_bits_proposal_opacity: int | List[int] | Callable[[int, torch.Tensor, torch.Tensor], List[int]] = None,
        n_bit_baselayer_scaling: int = None,
        n_bits_proposal_scaling: int | List[int] | Callable[[int, torch.Tensor, torch.Tensor], List[int]] = None,
        n_bit_baselayer_features_dc: int = None,
        n_bits_proposal_features_dc: int | List[int] | Callable[[int, torch.Tensor, torch.Tensor], List[int]] = None,
        n_bit_baselayer_features_rest: List[int] = [],
        n_bits_proposal_features_rest: List[int] | List[int] | List[Callable[[int, torch.Tensor, torch.Tensor], List[int]]] = [],
        **kwargs
    ):
        super().__init__(max_sh_degree=max_sh_degree, **kwargs)
        self.n_bit_baselayer = n_bit_baselayer
        self.n_bits_proposal_rotation_re = n_bits_proposal_rotation_re or n_bits_proposal
        self.n_bit_baselayer_rotation_re = n_bit_baselayer_rotation_re or n_bit_baselayer
        self.n_bits_proposal_rotation_im = n_bits_proposal_rotation_im or n_bits_proposal
        self.n_bit_baselayer_rotation_im = n_bit_baselayer_rotation_im or n_bit_baselayer
        self.n_bits_proposal_opacity = n_bits_proposal_opacity or n_bits_proposal
        self.n_bit_baselayer_opacity = n_bit_baselayer_opacity or n_bit_baselayer
        self.n_bits_proposal_scaling = n_bits_proposal_scaling or n_bits_proposal
        self.n_bit_baselayer_scaling = n_bit_baselayer_scaling or n_bit_baselayer
        self.n_bits_proposal_features_dc = n_bits_proposal_features_dc or n_bits_proposal
        self.n_bit_baselayer_features_dc = n_bit_baselayer_features_dc or n_bit_baselayer
        self.n_bits_proposal_features_rest = [((n_bits_proposal_features_rest[i] or n_bits_proposal) if len(n_bits_proposal_features_rest) > i else n_bits_proposal) for i in range(max_sh_degree)]
        self.n_bit_baselayer_features_rest = [((n_bit_baselayer_features_rest[i] or n_bit_baselayer) if len(n_bit_baselayer_features_rest) > i else n_bit_baselayer) for i in range(max_sh_degree)]

    def encode_layers(self, values: torch.Tensor, ids: torch.Tensor, codebook: torch.Tensor, n_bit_baselayer: int, n_bits_proposal: int | List[int] | Callable[[int, torch.Tensor, torch.Tensor], List[int]]):
        # return encode_layers(values, ids, codebook, n_bit_baselayer, n_bits_proposal, visualize=values.shape[1] == 3)  # debug
        return encode_layers(values, ids, codebook, n_bit_baselayer, n_bits_proposal)

    def extract_layers(self, layers: List[Layer]):
        return extract_layers(layers)

    def encode_layers_exclude_zero(self, values: torch.Tensor, ids: torch.Tensor, codebook: torch.Tensor, n_bit_baselayer: int, n_bits_proposal: int | List[int] | Callable[[int, torch.Tensor, torch.Tensor], List[int]]):
        zeros_mask = ids == 0
        nonzero_values, nonzero_ids = values[~zeros_mask], ids[~zeros_mask]
        nonzero_codebook = codebook[nonzero_ids.unique()]
        nonzero_ids -= 1
        # assert (nonzero_codebook[nonzero_ids] == codebook[ids][~zeros_mask]).all() # debug
        # layers = encode_layers(nonzero_values, nonzero_ids, nonzero_codebook, n_bit_baselayer, n_bits_proposal, visualize=values.shape[1] == 3)  # debug
        layers = encode_layers(nonzero_values, nonzero_ids, nonzero_codebook, n_bit_baselayer, n_bits_proposal)
        return [expand_base_layer(layers[0], zeros_mask)] + layers[1:]

    def extract_layers_exclude_zero(self, layers: List[Layer]):
        layer, zero_mask = shrink_base_layer(layers[0])
        nonzero_ids, nonzero_codebook = extract_layers([layer] + layers[1:])
        ids = torch.zeros(zero_mask.shape[0], dtype=nonzero_ids.dtype, device=nonzero_ids.device)
        ids[~zero_mask] = nonzero_ids + 1
        codebook = torch.cat([torch.zeros((1, *nonzero_codebook.shape[1:]), dtype=nonzero_codebook.dtype, device=nonzero_codebook.device), nonzero_codebook])
        return ids, codebook

    def cluster2layers_features_dc(self, model: GaussianModel, ids, codebook):
        return self.encode_layers(model._features_dc.detach().squeeze(1), ids, codebook, self.n_bit_baselayer_features_dc, self.n_bits_proposal_features_dc)

    def layers2cluster_features_dc(self, layers: List[Layer]):
        ids, codebook = self.extract_layers(layers)
        return ids.unsqueeze(1), codebook

    def cluster2layers_features_rest(self, model: GaussianModel, sh_degree, ids, codebook):
        if codebook.shape[0] <= 1:  # all zero from ExcludeZeroQuantizer.generate_codebook
            return []
        sh_idx_start, sh_idx_end = (sh_degree + 1) ** 2 - 1, (sh_degree + 2) ** 2 - 1
        features_rest_flatten = model._features_rest.detach().transpose(1, 2).flatten(0, 1)
        features_rest = features_rest_flatten[:, sh_idx_start:sh_idx_end]
        ids_reshaped = ids.reshape(ids.shape[0] * 3)
        layers = self.encode_layers_exclude_zero(features_rest, ids_reshaped, codebook, self.n_bit_baselayer_features_rest[sh_degree], self.n_bits_proposal_features_rest[sh_degree])
        return layers

    def layers2cluster_features_rest(self, sh_degree: int, layers: List[Layer], reference_ids: torch.Tensor, reference_codebook: torch.Tensor):
        if len(layers) <= 0:  # all zero from ExcludeZeroQuantizer.generate_codebook
            sh_idx_start, sh_idx_end = (sh_degree + 1) ** 2 - 1, (sh_degree + 2) ** 2 - 1
            ids = torch.zeros((reference_ids.shape[0], 3), dtype=reference_ids.dtype, device=reference_ids.device)
            codebook = torch.zeros((1, sh_idx_end - sh_idx_start), dtype=reference_codebook.dtype, device=reference_codebook.device)
            return ids, codebook
        ids_reshaped, codebook = self.extract_layers_exclude_zero(layers)
        ids = ids_reshaped.reshape(reference_ids.shape[0], 3)
        return ids, codebook

    def cluster2layers_rotation_re(self, model: GaussianModel, ids, codebook):
        return self.encode_layers(model.get_rotation.detach()[:, 0:1], ids, codebook, self.n_bit_baselayer_rotation_re, self.n_bits_proposal_rotation_re)

    def cluster2layers_rotation_im(self, model: GaussianModel, ids, codebook):
        return self.encode_layers(model.get_rotation.detach()[:, 1:], ids, codebook, self.n_bit_baselayer_rotation_im, self.n_bits_proposal_rotation_im)

    def cluster2layers_opacity(self, model: GaussianModel, ids, codebook):
        return self.encode_layers(model._opacity.detach(), ids, codebook, self.n_bit_baselayer_opacity, self.n_bits_proposal_opacity)

    def cluster2layers_scaling(self, model: GaussianModel, ids, codebook):
        return self.encode_layers(model._scaling.detach(), ids, codebook, self.n_bit_baselayer_scaling, self.n_bits_proposal_scaling)

    def cluster2layers(self, model: GaussianModel, codebook_dict: Dict[str, torch.Tensor], ids_dict: Dict[str, torch.Tensor]):
        layers_dict: Dict[str, List[Layer]] = {}

        layers_dict["features_dc"] = self.cluster2layers_features_dc(model, ids_dict["features_dc"].squeeze(1), codebook_dict["features_dc"])
        for sh_degree in range(model.max_sh_degree):
            layers_dict[f'features_rest_{sh_degree}'] = self.cluster2layers_features_rest(model, sh_degree, ids_dict[f'features_rest_{sh_degree}'], codebook_dict[f'features_rest_{sh_degree}'])

        layers_dict["rotation_re"] = self.cluster2layers_rotation_re(model, ids_dict["rotation_re"], codebook_dict["rotation_re"])
        layers_dict["rotation_im"] = self.cluster2layers_rotation_im(model, ids_dict["rotation_im"], codebook_dict["rotation_im"])

        layers_dict["opacity"] = self.cluster2layers_opacity(model, ids_dict["opacity"], codebook_dict["opacity"])
        layers_dict["scaling"] = self.cluster2layers_scaling(model, ids_dict["scaling"], codebook_dict["scaling"])
        return layers_dict

    def layers2cluster(self, max_sh_degree: int, layers_dict: Dict[str, List[Layer]]):
        ids_dict: Dict[str, torch.Tensor] = {}
        codebook_dict: Dict[str, torch.Tensor] = {}
        ids_dict["features_dc"], codebook_dict["features_dc"] = self.layers2cluster_features_dc(layers_dict["features_dc"])
        for sh_degree in range(max_sh_degree):
            ids_dict[f'features_rest_{sh_degree}'], codebook_dict[f'features_rest_{sh_degree}'] = self.layers2cluster_features_rest(
                sh_degree, layers_dict[f'features_rest_{sh_degree}'], ids_dict["features_dc"], codebook_dict["features_dc"]
            )

        ids_dict["rotation_re"], codebook_dict["rotation_re"] = self.extract_layers(layers_dict["rotation_re"])
        ids_dict["rotation_im"], codebook_dict["rotation_im"] = self.extract_layers(layers_dict["rotation_im"])

        ids_dict["opacity"], codebook_dict["opacity"] = self.extract_layers(layers_dict["opacity"])
        ids_dict["scaling"], codebook_dict["scaling"] = self.extract_layers(layers_dict["scaling"])
        return codebook_dict, ids_dict

    def baselayer_ply_dtype(self, max_sh_degree: int, layers_dict: Dict[str, List[Layer]]):
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('rot_re', self.force_code_dtype or f"u{math.ceil(layers_dict['rotation_re'][0].n_bit / 8)}"),
            ('rot_im', self.force_code_dtype or f"u{math.ceil(layers_dict['rotation_im'][0].n_bit / 8)}"),
            ('opacity', self.force_code_dtype or f"u{math.ceil(layers_dict['opacity'][0].n_bit / 8)}"),
            ('scale', self.force_code_dtype or f"u{math.ceil(layers_dict['scaling'][0].n_bit / 8)}"),
            ('f_dc', self.force_code_dtype or f"u{math.ceil(layers_dict['features_dc'][0].n_bit / 8)}"),
        ]
        for sh_degree in range(max_sh_degree):
            if len(layers_dict[f"features_rest_{sh_degree}"]) <= 0:
                continue
            force_code_dtype = self.force_code_dtype or f"u{math.ceil(layers_dict[f'features_rest_{sh_degree}'][0].n_bit / 8)}"
            dtype_full.extend([
                (f'f_rest_{sh_degree}_0', force_code_dtype),
                (f'f_rest_{sh_degree}_1', force_code_dtype),
                (f'f_rest_{sh_degree}_2', force_code_dtype),
            ])
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
                continue
            features_rest = layers_dict[f'features_rest_{sh_degree}'][0].codes.reshape(-1, 3).cpu().numpy()
            data_full.extend(np.array_split(features_rest, 3, axis=1))
        return data_full

    def save_baselayer_ply(self, model: GaussianModel, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        dtype_full = self.baselayer_ply_dtype(model.max_sh_degree, layers_dict)
        data_full = self.baselayer_ply_data(model, layers_dict)

        elements = np.rec.fromarrays([data.squeeze(-1) for data in data_full], dtype=dtype_full)
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(ply_path)

    def save_baselayer_codebook(self, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        # save base layer codebooks
        codebooks = {f"{key}_codebook": layers[0].codebook.cpu().numpy() for key, layers in layers_dict.items() if len(layers) > 0}
        cluster_centers = {f"{key}_cluster_centers": layers[0].cluster_centers.cpu().numpy() for key, layers in layers_dict.items() if len(layers) > 0}
        n_bits = {f"{key}_n_bits": layers[0].n_bit for key, layers in layers_dict.items() if len(layers) > 0}
        n_leafs = {f"{key}_n_leafs": layers[0].n_leaf for key, layers in layers_dict.items() if len(layers) > 0}
        np.savez_compressed(os.path.splitext(ply_path)[0] + ".codebook.npz", **codebooks, **cluster_centers, **n_bits, **n_leafs)

    def save_baselayer(self, model: GaussianModel, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        self.save_baselayer_ply(model, ply_path, layers_dict)
        self.save_baselayer_codebook(ply_path, layers_dict)

    def save_enhencementlayer(self, layer: Layer, path: str):
        np.savez_compressed(
            path,
            codes=layer.codes.cpu().numpy(),
            codebook=layer.codebook.cpu().numpy(),
            cluster_centers=layer.cluster_centers.cpu().numpy(),
            n_bit=layer.n_bit,
            n_leaf=layer.n_leaf)

    def save_enhencementlayers(self, ply_path: str, layers_dict: Dict[str, List[Layer]]):
        for key, layers in layers_dict.items():
            if len(layers) <= 1:
                continue
            for i, layer in enumerate(layers[1:]):
                self.save_enhencementlayer(layer, os.path.splitext(ply_path)[0] + f".layer.{key}.{i + 1}.npz")

    def save_quantized(self, model: GaussianModel, ply_path: str):
        if self._codebook_dict == {}:
            codebook_dict, ids_dict = self.produce_clusters(model, self._codebook_dict)
            self._codebook_dict = codebook_dict
        else:
            ids_dict = self.find_nearest_cluster_id(model, self._codebook_dict)
            codebook_dict = self._codebook_dict
        layers_dict = self.cluster2layers(model, codebook_dict, ids_dict)
        self.save_baselayer(model, ply_path, layers_dict)
        self.save_enhencementlayers(ply_path, layers_dict)
        # ids_dict_orig = ids_dict
        # codebook_dict, ids_dict = self.layers2cluster(model, layers_dict)
        # for key in layers_dict.keys():
        #     if len(layers_dict[key]) <= 0:
        #         continue
        #     print(key, (self._codebook_dict[key][ids_dict_orig[key]] - codebook_dict[key][ids_dict[key]]).abs().max())
        return self.apply_clustering(model, codebook_dict, ids_dict)

    def load_baselayer_attr(self, key, codes, codebooks, device):
        return Layer(
            codes=codes,
            codebook=torch.tensor(codebooks[f"{key}_codebook"], device=device),
            cluster_centers=torch.tensor(codebooks[f"{key}_cluster_centers"], device=device),
            n_bit=codebooks[f"{key}_n_bits"].item(),
            n_leaf=codebooks[f"{key}_n_leafs"].item(),
        )

    def load_baselayer(self, max_sh_degree: int, ply_path: str, device):
        plydata = PlyData.read(ply_path)
        codebooks = np.load(os.path.splitext(ply_path)[0] + ".codebook.npz")

        layers_dict = {}
        elements = plydata['vertex']
        layers_dict["rotation_re"] = [self.load_baselayer_attr("rotation_re", torch.tensor(elements["rot_re"].copy(), dtype=torch.int64, device=device), codebooks=codebooks, device=device)]
        layers_dict["rotation_im"] = [self.load_baselayer_attr("rotation_im", torch.tensor(elements["rot_im"].copy(), dtype=torch.int64, device=device), codebooks=codebooks, device=device)]
        layers_dict["opacity"] = [self.load_baselayer_attr("opacity", torch.tensor(elements["opacity"].copy(), dtype=torch.int64, device=device), codebooks=codebooks, device=device)]
        layers_dict["scaling"] = [self.load_baselayer_attr("scaling", torch.tensor(elements["scale"].copy(), dtype=torch.int64, device=device), codebooks=codebooks, device=device)]
        layers_dict["features_dc"] = [self.load_baselayer_attr("features_dc", torch.tensor(elements["f_dc"].copy(), dtype=torch.int64, device=device), codebooks=codebooks, device=device)]
        for sh_degree in range(max_sh_degree):
            if not set(f'f_rest_{sh_degree}_{ch}' for ch in range(3)).issubset(prop.name for prop in elements.properties):
                layers_dict[f'features_rest_{sh_degree}'] = []
                continue
            features_rest = torch.tensor(np.stack([elements[f'f_rest_{sh_degree}_{ch}'] for ch in range(3)], axis=1), dtype=torch.int64, device=device)
            layers_dict[f'features_rest_{sh_degree}'] = [self.load_baselayer_attr(f'features_rest_{sh_degree}', features_rest.reshape(-1), codebooks=codebooks, device=device)]
        return layers_dict

    def load_enhencementlayer(self, ply_path: str, key: str, device):
        i = 0
        layers = []
        while os.path.exists(os.path.splitext(ply_path)[0] + f".layer.{key}.{i + 1}.npz"):
            layer = np.load(os.path.splitext(ply_path)[0] + f".layer.{key}.{i + 1}.npz")
            layers.append(Layer(
                codes=torch.tensor(layer["codes"], device=device),
                codebook=torch.tensor(layer["codebook"], device=device),
                cluster_centers=torch.tensor(layer["cluster_centers"], device=device),
                n_bit=layer["n_bit"].item(),
                n_leaf=layer["n_leaf"].item(),
            ))
            i += 1
        return layers

    def load_enhencementlayers(self, ply_path: str, layers_dict: Dict[str, List[Layer]], device):
        for key in layers_dict.keys():
            if len(layers_dict[key]) <= 0:
                continue
            layers_dict[key].extend(self.load_enhencementlayer(ply_path, key, device))
        return layers_dict

    def load_quantized(self, model: GaussianModel, ply_path: str):
        device = model._xyz.device
        layers_dict = self.load_baselayer(model.max_sh_degree, ply_path, device)
        layers_dict = self.load_enhencementlayers(ply_path, layers_dict, device)
        codebook_dict, ids_dict = self.layers2cluster(model.max_sh_degree, layers_dict)
        return self.apply_clustering(model, codebook_dict, ids_dict)
