import argparse
import torch
import os
from torch import nn
from PIL import Image
from torchvision import transforms

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

from utils import tensor_transfos

import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--input_image", type=str, help="path to the image that should be loaded")
    parser.add_argument("--input_mask", type=str, help="path to the true mask corresponding to the input image")
    parser.add_argument("--output_path", type=str, help="path to the folder where the results should be saved")
    parser.add_argument("--model_path", type=str, help="path to the folder where the modele to predict the segmentation map is saved")

    opt = parser.parse_args()

    model = train.createDeepLabv3(opt.model_path)

    model.to(opt.device)

    image, seg_mask = tensor_transfos.draw_seg_map_from_single_image(opt.input_image, opt.device, model)

#### FOREGROUND (transparent seg map) works but the background somehow does not work!!! Try out how overlay works!
    transparent_overlay_pred = tensor_transfos.create_transparent_overlay(image, seg_mask)

    input_mask = Image.open(opt.input_mask)

    transparent_overlay_true = tensor_transfos.create_transparent_overlay(image, input_mask)

    basename = tensor_transfos.create_basename_from_input_path(opt.input_image)

    transparent_overlay_pred.save(os.path.join(opt.output_path, f"predicted_{basename}_transp_mask.png"))

    transparent_overlay_true.save(os.path.join(opt.output_path, f"true_{basename}_transp_mask.png"))
