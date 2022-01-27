import os
import torch
import numpy as np
import pandas as pd
import os.path as osp
import shutil

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class initialize_dataset:
    def __init__(self, config, batch_csv):
        self.config = config
        self.dataset_params = config["dataset_params"]
        self.dataset_root = self.dataset_params["root_dir"]
        self.dataset = self.dataset_params["use"]
        self.dataset_base = osp.join(os.getcwd(), self.dataset_root, self.dataset)
        if batch_csv is not None:
            csv = batch_csv
        else:
            csv = self.dataset_params[self.dataset]["csv_filename"]
        self.dataset_file = osp.join(self.dataset_base,"batch-info", csv)
        self.images_dir = self.dataset_params[self.dataset]["images_dir"]

    def create_dataset_csv(self):
        if osp.isfile(self.dataset_file):
            return
        df = pd.DataFrame(os.listdir(self.images_dir))
        df.to_csv(self.dataset_file, index=False, header=False)


class custom_dataset(Dataset):
    def __init__(self, dataset_file, images_dir, img_resolution, outdir):
        self.images_df = pd.read_csv(dataset_file, index_col=False, header=None)
        self.images_dir = images_dir
        self.img_resolution = img_resolution
        self.outdir = outdir

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):

        image_name = self.images_df.iat[index, 0]
        image = osp.join(self.images_dir, image_name)

        name_arr = image_name.split(".")
        target_image = osp.join(self.outdir, name_arr[0], "target." + name_arr[1])
        if not osp.isfile(target_image):
            os.makedirs(osp.join(self.outdir, name_arr[0]), exist_ok=True)
            shutil.copy2(image, target_image)

        target_pil = Image.open(image).convert("RGB")
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(
            ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
        )
        target_pil = target_pil.resize(
            (self.img_resolution, self.img_resolution), Image.LANCZOS
        )
        target_uint8 = np.array(target_pil, dtype=np.uint8)

        return (
            torch.tensor(
                target_uint8.transpose([2, 0, 1]), device=torch.device("cuda")
            ),
            image_name,
        )


def get_dataloader(config, batch_csv, outdir):
    init_dataset = initialize_dataset(config, batch_csv)
    init_dataset.create_dataset_csv()
    gen = config["projector_params"]["general"]["use_generator"]
    gen_resolution = config["projector_params"]["generators"][gen]["img_resolution"]
    dataset = custom_dataset(
        init_dataset.dataset_file, init_dataset.images_dir, gen_resolution, outdir
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=config["projector_params"]["general"]["batch_size"],
    )
    return dataloader
