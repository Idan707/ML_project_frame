import os
import glob
import torch

import numpy as np
import pandas as pd

from PIL import Image
from PIL import ImageFile

from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms

from albumentations import (
    Compose,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset:
    def __init__(
        self,
        image_paths,
        targets,
        resize=None,
        augmentions=None
    ):

        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentions = augmentions

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = image.convert("RGB")
        targets = self.targets[item]

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]),
                resample=Image.BILINEAR
            )
            image = np.array(image)

            if self.augmentions is not None:
                augmented = self.augmentions(image=image)
                image = augmented["image"]

            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

            return {
                "image": torch.tensor(image, dtype=torch.float),
                "targets": torch.tensor(targets, dtype=torch.long)
            }



class SIIMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_ids,
        transform=True,
        preprocessing_fn=True
    ):
        """
        Dataset class for segmentation problem
        :param image_ids: ids of the images, list
        :param transform: True/False, no transform in validation
        :param preprocessing_fn: a function for preprocessing image        
        """
        self.data = defaultdict(dict)
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn
        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10, p=0.8
                ),
                OneOf(
                    [
                        RandomGamma(
                            gamma_limit=(90, 110)
                        ),
                        RandomBrightnessContrast(
                            brightness_limit=0.1,
                            contrast_limit=0.1
                        ),
                    ],
                    p=0.5,
                ),
            ]
        )

        for imgid in image_ids:
            files = glob.glob(os.path.join(TRAIN_PATH, imgid, "*.png"))
            self.data[counter] = {
                "img_path": os.path.join(
                    TRAIN_PATH, imgid + ".png"
                ),
                "mask_path": os.path.join(
                    TRAIN_PATH, imgid + "_mask.png"
                )
            }

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            img_path = self.data[item]["img_path"]
            mask_path = self.data[item]["mask_path"]

            img = Image.open(img_path)
            img = img.convert("RGB")

            img = np.transpose(img)
            mask = Image.open(mask_path)

            mask = (mask >= 1).astype("float32")

            if self.transform is True:
                augmented = self.aug(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]

            img = self.preprocessing_fn(img)

            return {
                "image": transforms.ToTensor()(img),
                "mask": transforms.ToTensor()(mask).float(),
            }    
