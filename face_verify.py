import os
import os.path as osp
import glob

import cv2
from deepface import DeepFace
from tqdm import tqdm

import itertools
import click

import yaml
from csv import DictWriter
import csv

def verify_face(config, sheet, img_folder, target_img, projected_imgs):

    img1 = cv2.imread(target_img)
    headers = config["face_verify_params"]["csv_headers"]

    if not osp.exists(sheet):
        with open(sheet, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    model_dict = {}
    models = config["face_verify_params"]["models"]

    for model_name in models:
        model_dict[model_name] = DeepFace.build_model(model_name)

    face_detectors = config["face_verify_params"]["detectors"]
    metrics = config["face_verify_params"]["metrics"]

    data = []
    for proj_img in projected_imgs:
        img2 = cv2.imread(proj_img)
        reg_noise_wgt = proj_img.rsplit("/", 2)[1]
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
                )
            except Exception as e:
                print(e)
                print(
                    "For combination: " + tuple[0] +
                    ", " + tuple[1] + ", " + tuple[2]
                )
            result["target"] = img_folder
            result["detector"] = tuple[1]
            result["reg_loss_weight"] = reg_noise_wgt
            data.append(result)

    with open(sheet, "a", encoding="UTF8", newline="") as f:
        dictwriter = DictWriter(f, fieldnames=headers)
        dictwriter.writerows(data)
        f.close()

    print("\nModified master-report for " + img_folder)


@click.command()
@click.option(
    "--config",
    "config_path",
    help="path to config",
    default="./config/default.yml",
    type=str,
    required=True,
)
# Here the outdir = /nas/vista-hdd01/users/raidurga/outputs/ai2ai_run_005/expts/batch-001
@click.option("--outdir", help="Source File", type=str, required=True)
def verify_batch(config_path: str, outdir: str):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    sheet = osp.join(outdir.rsplit("/", 2)
                     [0], config["face_verify_params"]["csv"])

    for img_folder in tqdm(os.scandir(outdir)):
        if img_folder.is_dir():
            target_img = glob.glob(
                        img_folder.path + "/target.*")[0]
            projected_imgs = glob.glob(img_folder.path + "/*/projected.*")
            if projected_imgs:
                verify_face(
                    config,
                    sheet,
                    img_folder.name,
                    target_img,
                    projected_imgs,
                )


@click.command()
@click.option(
    "--config",
    "config_path",
    help="path to config",
    default="./config/default.yml",
    type=str,
    required=True,
)
# Here the outdir = /nas/vista-hdd01/users/raidurga/outputs/ai2ai_run_005/expts
@click.option("--outdir", help="Source File", type=str, required=True)
def verify_all(config_path: str, outdir: str):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    sheet = osp.join(outdir, config["face_verify_params"]["csv"])

    for img_folder in tqdm(os.scandir(outdir)):
        if img_folder.is_dir():
            print(f"Current Image Path is: '{img_folder.path}'")
            target_img = glob.glob(
                img_folder.path + "/target.*")[0]
            projected_imgs = glob.glob(
                img_folder.path + "/*/projected.*")

            if projected_imgs:
                verify_face(
                    config,
                    sheet,
                    img_folder.name,
                    target_img,
                    projected_imgs,
                )

complete_verify = True

if __name__ == "__main__":

    if complete_verify:
        verify_all()
    else:
        verify_batch()