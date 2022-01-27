import os
import glob
import cv2

import itertools
import click
import yaml
from tqdm import tqdm
from deepface import DeepFace


def custom_preprocess(config_path: str, img: str):

    with open(config_path) as f:
        config = yaml.load(f)
    model_dict = {}
    models = config["face_verify_params"]["models"]

    for model_name in models:
        model_dict[model_name] = DeepFace.build_model(model_name)

    for tuple in itertools.product(models, config["face_verify_params"]["detectors"]):
        try:

            DeepFace.custom_preprocess(
                img,
                model_name=tuple[0],
                model=model_dict[tuple[0]],
                detector_backend=tuple[1],
                #  detector=detectors_dict[tuple[1]],
                #  detector=None,
                align=True,
                enforce_detection=True,
                normalization="base",
            )

        except Exception as e:
            print(e)
            print("For combination: " + tuple[0] + ", " + tuple[1])


preprocess_images = True
img_dirs = [
    "img_001"
    # "img_002",
    # "img_003",
    # "img_004",
    # "img_005",
    # "img_006",
    # "img_007",
    # "img_008",
    # "img_009",
    # "img_0010",
]


@click.command()
@click.option(
    "--config",
    "config_path",
    help="path to config",
    default="./config/default.yaml",
    type=str,
    required=False,
)
@click.option("--outdir", help="Source Path", type=str, default=None, required=False)
@click.option("--img_path", help="Source File", type=str, default=None, required=False)
def execute_main(config_path: str, outdir: str, img_path: str):

    if preprocess_images:
        if outdir is not None:
            for imgdir in img_dirs:
                file_list = glob.glob(outdir + imgdir + "/*/projected.png")
                pbar = tqdm(file_list)
                for file_path in pbar:
                    custom_preprocess(config_path, file_path)
        elif img_path is not None:
            custom_preprocess(config_path, img_path)


if __name__ == "__main__":
    execute_main()
