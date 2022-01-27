import subprocess
import csv
import uuid as UUID
import os
import sys
import yaml
import os.path as osp

dataset_root = "/nas/vista-ssd01/users/mehussein/aifi/dataset"

def create_dataset_dir():
    uuid = str(UUID.uuid4())
    print("Please save the below unique id (UUID) generated for this request.")
    print(uuid)
    output_dir = osp.join(dataset_root, uuid)
    return uuid, output_dir


batch_size = 5
uuid, dataset_dir = create_dataset_dir()
batch_info_path = osp.join(dataset_dir, "batch-info")
outdir = osp.join(dataset_dir, "execution")

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(batch_info_path, exist_ok=True)
os.makedirs(outdir, exist_ok=True)


def create_batch_files(images):
    batch_file_names = []
    for batch_idx, image_start_idx in enumerate(range(0, len(images), batch_size)):
        batch_file = osp.join(batch_info_path, "batch-" + str(batch_idx + 1) + ".csv")
        batch_file_names.append(batch_file)
        image_end_idx = min(image_start_idx + batch_size, len(images))
        print(f"batch_file {batch_file} has images {image_start_idx}:{image_end_idx}")
        with open(batch_file, "a", encoding="UTF8") as f:
            writer = csv.writer(f)
            for image in images[image_start_idx:image_end_idx]:
                writer.writerow([image])
    return batch_file_names


def create_config():
    with open("./config/dynamic.yml") as f:
        doc = yaml.safe_load(f)
    dataset_params = doc["dataset_params"]
    dataset_params["use"] = uuid
    dataset_params[uuid] = dataset_params["dynamic"]
    del dataset_params["dynamic"]

    new_path = osp.join(dataset_dir, "config.yml")
    with open(new_path, "w") as f:
        yaml.dump(doc, f)
    return new_path


if __name__ == "__main__":

    images = sys.argv
    images.pop(0)
    dataset_csv_files = create_batch_files(images)
    print(dataset_csv_files)
    config_path = create_config()

    for dataset_csv in dataset_csv_files:
        print(f"Projection of images in file {dataset_csv}")
        print(f"=============================" + '=' * len(dataset_csv))
        cmd = (
            "/nas/vista-ssd01/users/mehussein/miniconda3/envs/aifi/bin/python run_projector.py"
            + " --outdir "
            + outdir
            + " --dataset_csv "
            + dataset_csv
            + " --config_path "
            + config_path
            # + "; conda deactivate"
        )
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        print("\n-- Projection Errors ...\n\n" + str(err))

        print(f"Verification of images in file {dataset_csv}")
        print(f"===============================" + '=' * len(dataset_csv))

        cmd = (
            "/nas/vista-ssd01/users/mehussein/miniconda3/envs/deepface/bin/python face_verify.py"
            + " --outdir "
            + outdir
            + " --config "
            + config_path
        )
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        print("\n-- Verification Errors ...\n\n" + str(err))
