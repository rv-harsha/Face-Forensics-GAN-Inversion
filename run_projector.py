import yaml
import torch
import argparse
from dataset import get_dataloader
from projector import projector


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GAN Inversion Analysis")
    parser.add_argument("--outdir", dest="outdir", type=str, help="Output Directory")
    parser.add_argument(
        "--dataset_csv", dest="dataset_csv", type=str, help="Dataset batch csv"
    )
    parser.add_argument(
        "--config_path", dest="config_path", type=str, help="Dataset config path"
    )
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda")
    
    dataloader = get_dataloader(config, args.dataset_csv, args.outdir)
    for reg_wgt in config["noise_reg_wgts"]:
        proj = projector(config, reg_wgt, device, args.outdir)
        proj.project(dataloader)