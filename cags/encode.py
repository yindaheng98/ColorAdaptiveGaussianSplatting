import os
import shutil
import torch
from gaussian_splatting import GaussianModel
from cags.quantization import ScalableQuantizer, DracoCompressedScalableQuantizer
from scalablevq import n_bits_proposal_balanced_clusters, n_bits_proposal_balanced_values
from cags.tiling import MortonTiling, AverageSplitTiling
from cags.tilequant import TillingScalableQuantizer
from cags.interframe import NoInterframeExtractor, InterframeExtractor, QuantizedInterframeExtractor
from cags.codec import Codec


def copy_not_exists(source, destination):
    if os.path.exists(destination):
        if os.path.samefile(source, destination):
            return
        os.remove(destination)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy(source, destination)


def encode_once(codec: Codec, source, destination, iteration, sh_degree, device, init):
    copy_not_exists(os.path.join(source, "cfg_args"), os.path.join(destination, "cfg_args"))
    copy_not_exists(os.path.join(source, "cameras.json"), os.path.join(destination, "cameras.json"))
    input = os.path.join(source, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
    output = os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
    gaussians = GaussianModel(sh_degree).to(device)
    gaussians.load_ply(input)
    shutil.rmtree(os.path.join(destination, "point_cloud", "iteration_" + str(iteration)), ignore_errors=True)
    os.makedirs(os.path.join(destination, "point_cloud", "iteration_" + str(iteration)), exist_ok=True)
    if init:
        codec.encode_init(gaussians, output)
    else:
        codec.encode_next(gaussians, output)


def decode_once(codec: Codec, destination, iteration, sh_degree, device, init):
    output = os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
    gaussians = GaussianModel(sh_degree).to(device)
    if init:
        gaussians = codec.decode_init(gaussians, output)
    else:
        gaussians = codec.decode_next(gaussians, output)
    gaussians.save_ply(output)


def main(
        source, destination, iteration,
        source_init, destination_init, iteration_init,
        frame_format, start_frame, end_frame,
        sh_degree, device, draco, encode,
        tiling_first, tiling_rest, interframe,
        **kwargs):
    if draco:
        frame_quantizer = TillingScalableQuantizer(DracoCompressedScalableQuantizer(**kwargs), MortonTiling())
    else:
        frame_quantizer = TillingScalableQuantizer(ScalableQuantizer(**kwargs), MortonTiling())
    match interframe:
        case "none":
            frame_extractor = NoInterframeExtractor()
        case "quantize":
            frame_extractor = QuantizedInterframeExtractor(quantizer=frame_quantizer.quantizer)
        case "interframe":
            frame_extractor = InterframeExtractor()
        case _:
            raise ValueError(f"Unknown interframe option: {interframe}")
    codec = Codec(
        frame_extractor=frame_extractor,
        frame_quantizer=frame_quantizer,
        tiling_rest=AverageSplitTiling() if tiling_rest else None,
        tiling_first=tiling_first,
    )
    if encode:
        encode_once(
            codec,
            source=source_init, destination=destination_init, iteration=iteration_init,
            sh_degree=sh_degree, device=device, init=True
        )
    else:
        decode_once(
            codec,
            destination=destination_init, iteration=iteration_init,
            sh_degree=sh_degree, device=device, init=True
        )
    for i in range(start_frame, end_frame + 1):
        frame = frame_format % i
        frame_source = os.path.join(source, frame)
        frame_destination = os.path.join(destination, frame)
        if encode:
            encode_once(
                codec,
                source=frame_source, destination=frame_destination, iteration=iteration,
                sh_degree=sh_degree, device=device, init=False
            )
        else:
            decode_once(
                codec,
                destination=frame_destination, iteration=iteration,
                sh_degree=sh_degree, device=device, init=False
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", default=30000, type=int)
    parser.add_argument("--source_init", required=True, type=str)
    parser.add_argument("--destination_init", required=True, type=str)
    parser.add_argument("--iteration_init", default=30000, type=int)
    parser.add_argument("--frame_format", default="frame%d", type=str)
    parser.add_argument("--frame_start", required=True, type=int)
    parser.add_argument("--frame_end", required=True, type=int)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    parser.add_argument("--draco", action='store_true')
    parser.add_argument("--decode", action='store_true')
    parser.add_argument("--no_tiling_first", action='store_true')
    parser.add_argument("--no_tiling_rest", action='store_true')
    parser.add_argument("--interframe", choices=["none", "quantize", "interframe"], default="interframe")
    args = parser.parse_args()
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    main(
        source=args.source, destination=args.destination, iteration=args.iteration,
        source_init=args.source_init, destination_init=args.destination_init, iteration_init=args.iteration_init,
        frame_format=args.frame_format, start_frame=args.frame_start, end_frame=args.frame_end,
        sh_degree=args.sh_degree, device=args.device, draco=args.draco, encode=not args.decode,
        tiling_first=not args.no_tiling_first, tiling_rest=not args.no_tiling_rest, interframe=args.interframe,
        **configs)
