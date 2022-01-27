import os
import os.path as osp
import glob

import cv2
from deepface import DeepFace

import itertools
import pandas as pd
import click

import yaml

def verify_face(config, outdir):

    model_dict = {}
    models = config["face_verify_params"]["models"]

    for model_name in models:
        model_dict[model_name] = DeepFace.build_model(model_name)

    face_detectors = config["face_verify_params"]["detectors"]
    metrics = config["face_verify_params"]["metrics"]

    img1 = cv2.imread(outdir+'/batch-2/7/target.jpg')
    other_imgs = glob.glob(outdir + "/*/*/target.jpg")

    data = []
    for proj_img in other_imgs:
        img_folder = proj_img.rsplit("/", 2)[1]
        img2 = cv2.imread(proj_img)
        reg_noise_wgt = 'NA'
        for tuple in itertools.product(
            models,
            face_detectors,
            metrics,
        ):
            result = {}
            try:
                result = DeepFace.verify(
                    img1,
                    img2,
                    model_name=tuple[0],
                    model=model_dict[tuple[0]],
                    distance_metric=tuple[2],
                    detector_backend=tuple[1],
                    enforce_detection=True,
                    normalization="base",
                    prog_bar=True,
                )
            except Exception as e:
                print(e)
                print(
                    "For combination: " + tuple[0] + ", " + tuple[1] + ", " + tuple[2]
                )
            result["target"] = img_folder
            result["detector"] = tuple[1]
            result["reg_loss_weight"] = reg_noise_wgt
            data.append(result)
    return data


@click.command()
@click.option(
    "--config",
    "config_path",
    help="path to config",
    default="./config/default.yml",
    type=str,
    required=False,
)
@click.option("--outdir", help="Source File", type=str, required=False)
def execute_main(config_path: str, outdir: str):

    with open(config_path) as f:
        config = yaml.load(f)

    final_report = []
    outdir = '/nas/vista-hdd01/users/raidurga/outputs/ai2ai_run_004/expts'
    sheet = os.path.join(outdir, "random-report.xlsx")
    writer = pd.ExcelWriter(sheet, engine="openpyxl")
    final_report.extend(
        verify_face(
            config,
            outdir
        )
    )
    pd.DataFrame(final_report).to_excel(writer, sheet_name="Report", index=False)
    writer.save()

if __name__ == "__main__":
    execute_main()