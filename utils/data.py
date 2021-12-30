from numpy.lib.type_check import imag
import torch
import os
import numpy as np
from typing import Any, Callable, Optional

from torchvision.datasets.vision import VisionDataset
from PIL import Image
import torchvision.transforms.functional as TF
import random
from torchvision import transforms


def apply_transformations(image, mask):

        # image and mask are PIL image object. 
        img_w, img_h = image.size

        # Random horizontal flipping
        if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
        
        # Random affine
        affine_param = transforms.RandomAffine.get_params(
                degrees = [-180, 180], translate = [0.3,0.3],  
                img_size = [img_w, img_h], scale_ranges = [1, 1.3], 
                shears = [2,2])
        image = TF.affine(image, 
                        affine_param[0], affine_param[1],
                        affine_param[2], affine_param[3])
        mask = TF.affine(mask, 
                        affine_param[0], affine_param[1],
                        affine_param[2], affine_param[3])

        return image, mask


class SegmentationDataset(VisionDataset):
    """
    Dataset class for image segmentation task based on the torchvision Vision Dataset. Compatible with torchvision
    """

    def __init__(self,
                image_list: list,
                mask_list: list,
                train_transforms: bool,
                seed: int = None,
                image_color_mode: str = "grayscale",
                mask_color_mode: str = "grayscale") -> None:

        """
        Args:
            image_list: list of paths to the images
            mask_list: list of paths to the masks
            
            backend: the way the images are loaded - next to pil, one could add cv2
            train_transforms: boolean to select if transformation from the method transform are applied on dataset (set to None for validation)

            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'
            mask_color_mode (str, optional): 'rgb' or 'grayscale'

        Data Directory structure:
            --train_val
            ------images
            -----------Image1
            -----------ImageN
            ------masks
            -----------Mask1
            -----------MaskN
            --test
            ------images
            -----------Image1
            -----------ImageN
            ------masks
            -----------Mask1
            -----------MaskN
        """

        super().__init__(transforms)
        self.backend = "pil"
        self.train_transforms = train_transforms
        self.image_list = image_list
        self.mask_list = mask_list
        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        # Raises (OSError, ValueError) could be implemented here

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item: int):

        image = self.image_list[item]
        mask = self.mask_list[item]

        with open(image, 'rb') as image_file, open(mask, 'rb') as mask_file:

            if self.backend == "pil":
                    
                image = Image.open(image_file).convert("L")
                
                mask = Image.open(mask_file).convert("L")

                normalize = transforms.ToTensor()

                if self.train_transforms:
                    image, mask = apply_transformations(image, mask)

                image = normalize(image)
                #mask = normalize(mask)
                mask = np.array(mask)
                mask_B_1_H_W = np.array([mask])
                mask = torch.tensor(mask)
                mask_B_1_H_W = torch.tensor(mask_B_1_H_W)
                            
            else:
                raise Exception("Backend not implemented")

            return image, mask, mask_B_1_H_W