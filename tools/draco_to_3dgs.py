import argparse
import os
from gaussian_splatting import GaussianModel

parser = argparse.ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", required=True, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    gaussians.load_ply(load_ply)
    gaussians.save_ply(load_ply)
