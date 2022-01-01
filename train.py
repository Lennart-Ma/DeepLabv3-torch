import os
from cv2 import transform
from sklearn import metrics
import torch
from torch import nn
from torchvision import transforms

import argparse
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import albumentations

from utils.data import SegmentationDataset
from utils.training_loops import training_loop
from utils.training_loops import val_loop
from torchvision import models

from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from utils.losses import Generalized_Dice_Loss

def createDeepLabv3():
    """DeepLabv3 class with custom head
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True,
                                                    )
    
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    model.classifier = DeepLabHead(2048, num_classes=4) # num_classes is equal to output channels
    # Set the model in training mode
    # print(model)
    return model


def train(mean, std, fold, training_data_path, device, epochs, train_bs, val_bs, outdir, lr):

    df = pd.read_csv(os.path.join(training_data_path, "gt.csv"))
    mean = mean
    std = std

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True)
    print("train set length: ", len(df_train))
    print("val set length: ", len(df_val))

    model = createDeepLabv3()

    model.to(device)

    # Set up the train_loader and val_loader

    train_images = df_train.image.values.tolist()
    train_images = [os.path.join(training_data_path, "images", i) for i in train_images]
    train_masks = df_train.image.values.tolist()
    train_masks = [os.path.join(training_data_path, "masks", i) for i in train_masks]

    val_images = df_val.image.values.tolist()
    val_images = [os.path.join(training_data_path, "images", i) for i in val_images]
    val_masks = df_val.image.values.tolist()
    val_masks = [os.path.join(training_data_path, "masks", i) for i in val_masks]


    train_dataset = SegmentationDataset(
        train_images,
        train_masks,
        train_transforms=True)

    val_dataset = SegmentationDataset(
        val_images,
        val_masks,
        train_transforms=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=2
    )

    # Set loss function
    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.MSELoss(reduction='mean')
    #loss_function = Generalized_Dice_Loss()

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        threshold=0.0001,
        mode="max"
    )

    train_loss_all = []
    val_loss_list = []
    mean_dice_list = []
    mean_dice_exc_c_0_list = []

    # Start training
    for epoch in range(epochs):

        train_loss = training_loop(model, device, train_loader, optimizer, loss_function, epoch, num_epochs=epochs)

        val_loss, mean_dice, mean_dice_exc_c_0 = val_loop(model, device, val_loader, loss_function)
        
        scheduler.step(mean_dice)
    
        if all(mean_dice > i for i in mean_dice_list):
            torch.save(model.state_dict(), os.path.join(outdir, f"model_fold_{fold}.bin"))
            print("Model with improved mean_dice_list saved to outdir")

        train_loss_all.append(train_loss)
        val_loss_list.append(val_loss)
        mean_dice_list.append(mean_dice)
        mean_dice_exc_c_0_list.append(mean_dice_exc_c_0)


    train_loss_all = np.array(train_loss_all)
    train_loss_all = train_loss_all.flatten()
    train_loss_plot = plt.figure()
    plt.plot(train_loss_all)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    train_loss_plot.savefig(os.path.join(opt.outdir, f'training_loss_fold{opt.fold}.png'))


    val_loss_plot = plt.figure()
    plt.plot(val_loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    val_loss_plot.savefig(os.path.join(opt.outdir, f'validation_loss_fold{opt.fold}.png'))


    mean_dice_plot = plt.figure()
    plt.plot(mean_dice_list)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice")
    mean_dice_plot.savefig(os.path.join(opt.outdir, f'mean_dice_fold{opt.fold}.png'))

    mean_dice_exc_c_0_plot = plt.figure()
    plt.plot(mean_dice_exc_c_0_list)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice w/o channel 0")
    mean_dice_exc_c_0_plot.savefig(os.path.join(opt.outdir, f'mean_dice_without_ch0_fold{opt.fold}.png'))


    print("plots saved..")

    end_time = datetime.now()

    end_time = end_time.strftime("%H:%M:%S")
    print("Start Time = ", starting_time, "End Time = ", end_time)

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80, help="number of epochs")
    parser.add_argument("--train_batch", type=int, default=16, help="batch size for training")
    parser.add_argument("--val_batch", type=int, default=16, help="batch size for validation")
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--pretrained_own", type=str, help="path to a pretrained model (.bin)")
    parser.add_argument("--metric", type=str, default="f1_score", help="The metric on which to save improved models - (f1_score,auc, precision)")
    # Needed inputs:
    parser.add_argument("--fold", type=int, help="which fold is the val fold")
    parser.add_argument("--dir", type=str, help="path to the folder containing the images, mask and gt")
    parser.add_argument("--outdir", type=str, help="outputdir where the files are stored")

    opt = parser.parse_args()

    opt_dict = vars(opt)

    json_object = json.dumps(opt_dict)

    open(os.path.join(opt.outdir, f'training_options{opt.fold}.txt'), 'w').close()

    with open(os.path.join(opt.outdir, f'training_options{opt.fold}.txt'), 'a') as f:
        json.dump(json_object, f)
    
    starting_time = datetime.now()

    starting_time = starting_time.strftime("%H:%M:%S")
    print("Starting Time =", starting_time)

    #mean, std = get_mean_std(opt.dataset)

    mean, std = (0.2286, 0.2334028) #mean and std for the heart dataset

    train(mean, std, opt.fold, opt.dir, opt.device, opt.epochs, opt.train_batch, opt.val_batch, opt.outdir, opt.lr)