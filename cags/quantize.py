import os
import shutil
from gaussian_splatting import GaussianModel
from cags.quantization import ScalableQuantizer, DracoCompressedScalableQuantizer
from scalablevq import n_bits_proposal_balanced_clusters, n_bits_proposal_balanced_values


def copy_not_exists(source, destination):
    if os.path.exists(destination):
        if os.path.samefile(source, destination):
            return
        os.remove(destination)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy(source, destination)


def quantize(source, destination, iteration, sh_degree, device, quantize, draco, **kwargs):
    copy_not_exists(os.path.join(source, "cfg_args"), os.path.join(destination, "cfg_args"))
    copy_not_exists(os.path.join(source, "cameras.json"), os.path.join(destination, "cameras.json"))
    input = os.path.join(source, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
    output = os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud_quantized.ply")
    gaussians = GaussianModel(sh_degree).to(device)
    gaussians.load_ply(input)
    if draco:
        quantizer = DracoCompressedScalableQuantizer(**kwargs)
    else:
        quantizer = ScalableQuantizer(**kwargs)
    if quantize:
        shutil.rmtree(os.path.join(destination, "point_cloud", "iteration_" + str(iteration)), ignore_errors=True)
        os.makedirs(os.path.join(destination, "point_cloud", "iteration_" + str(iteration)), exist_ok=True)
        gaussians = quantizer.save_quantized(gaussians, output)
    else:
        gaussians = quantizer.load_quantized(gaussians, output)
        output = os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
        gaussians.save_ply(output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", default=30000, type=int)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    parser.add_argument("--draco", action='store_true')
    parser.add_argument("--dequantize", action='store_true')
    args = parser.parse_args()
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    quantize(
        source=args.source, destination=args.destination, iteration=args.iteration, sh_degree=args.sh_degree,
        device=args.device, quantize=not args.dequantize, draco=args.draco, **configs)
