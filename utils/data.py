import torch
import os
from typing import Any, Callable, Optional

from torchvision.datasets.vision import VisionDataset
from PIL import Image



class SegmentationDataset(VisionDataset):
    """
    Dataset class for image segmentation task based on the torchvision Vision Dataset. Compatible with torchvision
    """

    def __init__(self,
                root: str,
                image_folder: str,
                mask_folder: str,
                backend: str = "pil",
                transforms: Optional[Callable] = None,
                seed: int = None,
                image_color_mode: str = "grayscale",
                mask_color_mode: str = "grayscale") -> None:

        """
        Args:
            root (str): Root dir
            image_folder (str): Name of the image folder - located in root dir
            mask_folder (str): Name of the mask folder - located in root dir
            
            backend: the way the images are loaded - next to pil, one could add cv2
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.

            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'
            mask_color_mode (str, optional): 'rgb' or 'grayscale'

        Data Directory structure:
            --Train
            ------Image
            -----------Image1
            -----------ImageN
            ------Mask
            -----------Mask1
            -----------MaskN
            --Test
            ------Image
            -----------Image1
            -----------ImageN
            ------Mask
            -----------Mask1
            -----------MaskN
        """

        super().__init__(root, transforms)
        image_folder_path = os.path.join(root, image_folder)
        mask_folder_path = os.path.join(root, mask_folder)
        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        self.image_names = sorted(image_folder_path.glob("*"))
        self.mask_names = sorted(mask_folder_path.glob("*"))

        # Raises (OSError, ValueError) could be implemented here

    def __len__(self):
        return len(self.image_folder_path)

    def __getitem__(self, item: int):

        image_path = self.image_names[item]
        mask_path = self.mask_names[item]

        with open(image_path, 'rb') as image_file, open(mask_path, 'rb') as mask_file:


            if self.backend == "pil":
                    
                image = Image.open(image_file)

                if self.image_color_mode == "rgb":
                    image = image.convert("RGB")
                elif self.image_color_mode == "grayscale":
                    image = image.convert("L")
                mask = Image.open(mask_file)
                if self.mask_color_mode == "rgb":
                    mask = mask.convert("RGB")
                elif self.mask_color_mode == "grayscale":
                    mask = mask.convert("L")

                sample = {'image': image, 'mask': mask}
                if self.transforms:
                    sample["image"] = self.transforms(sample["image"])
                    sample["mask"] = self.transforms(sample["mask"])
            
            else:
                raise Exception("Backend not implemented")

            return sample