import torch

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn import functional as F

from utils import metrics
from utils import tensor_transfos


def training_loop(model, device, train_loader, optimizer, loss_function, epoch, num_epochs):

    train_loss = []

    model.train()

    n_total_steps = len(train_loader)

    for batch_idx, (data, mask, mask_B_1_H_W) in enumerate(train_loader):

        data = data.to(device)

        data = data.type(torch.cuda.FloatTensor)

        data = data.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            print(data.shape)

            output = model(data)

            #torch.set_printoptions(profile="full")

            mask = mask.type(torch.cuda.LongTensor)

            loss = loss_function(output['out'], mask)

            loss.backward()

            optimizer.step()

            if batch_idx % 2 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{n_total_steps}], Loss: {loss.item():.8f}')

            train_loss.append(loss.item())
        
    return train_loss


def val_loop(model, device, val_loader, loss_function):

    model.eval()

    with torch.no_grad():
        for i, (data, mask, mask_B_1_H_W) in enumerate(val_loader):
            
            dice_coeff = metrics.Dice_Coefficient()

            data = data.to(device)

            data = data.type(torch.cuda.FloatTensor)

            data = data.to(device)

            output = model(data)

            mask = mask.type(torch.cuda.LongTensor)

            curr_loss = loss_function(output['out'], mask)

            target = tensor_transfos.make_one_hot(mask_B_1_H_W, C=4)

            # outputs a tensor of shape (batch_size, n_classes(channels) + 2)
            # each row belongs to the dice values of one output['out'], target combination
            # the first n_classes columns belong the dice coeff of each channel
            # the pre last column [-2] belongs to the mean dice value over all channels/classes per image
            # the last column [-1] belongs to the mean dice value over all channels/classes except the first one per iamge
            dice_coeff_values = dice_coeff(output['out'], target)

            curr_mean_dice = torch.mean(dice_coeff_values[:, -2], dim=0)
            curr_mean_dice_exc_c_0 = torch.mean(dice_coeff_values[:, -1], dim=0)

            if i==0:
                loss = np.array([curr_loss.data.cpu().numpy()])
                mean_dice = np.array([curr_mean_dice.data.cpu().numpy()])
                mean_dice_exc_c_0 = np.array([curr_mean_dice_exc_c_0.data.cpu().numpy()])

            else:
                loss = np.concatenate((loss, np.array([curr_loss.data.cpu().numpy()])))
                mean_dice = np.concatenate((mean_dice, np.array([curr_mean_dice.data.cpu().numpy()])))
                mean_dice_exc_c_0 = np.concatenate((mean_dice_exc_c_0, np.array([curr_mean_dice_exc_c_0.data.cpu().numpy()])))

    
    print("Val loss: ", np.mean(loss), ", Mean dice: ", np.mean(mean_dice),
    ", Mean dice w/o ch 0: ", np.mean(mean_dice_exc_c_0))

    return np.mean(loss), np.mean(mean_dice), np.mean(mean_dice_exc_c_0)