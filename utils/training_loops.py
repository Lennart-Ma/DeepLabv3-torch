import torch

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn import functional as F

from utils import metrics


def seg_model_output_to_categorical(output, C):
        """
        Takes in the output of a DeepLabV3 model (multichannel output) with shape [B,C,H,W] (B:batch size, C:channel/equal to num_classes, H/W:height/width of the image)
        and returns a categorical vector of unique values (0,1,2,3,n)
        
        ARGS:
        output: predicted mask of model (DeepLabv3) with predicted values with shape [B,C,H,W] (B:batch size, C:channel/equal to num_classes, H/W:height/width of the image)
        where C can have different arbitrary values
        num_classes: int
        
        PROCEDURE:
        1. takes in output (shape: (16,4,256,256) with many different values) 
        2. apply softmax to get values which add up to 1 along each channel (highest value would be equal to the class) (shape: (16,4,256,256) with values which add up to 1 along channel axis)
        3. apply argmax along dim=1 to get a tensor of shape (16,256,256) where each value tells the index of the predicted class for each value
        4. reshape to get a tensor of shape (16,1,256,256) where each value tells the index of the predicted class for each value
        5. apply make_one_hot(x, num_classes) to get a tensor of shape (16,4,256,256) where one of the 4 channels (axis 1) has a value of 1, the other 3 channels a value of 0  
        
        RETURNS:
        categorical tensor: a tensor of shape [B,C,H,W] (B:batch size, C:channel/equal to num_classes, H/W:height/width of the image)
        where C has unique values [0,1] and only one 1 along the axis
        """
        torch.set_printoptions(profile="full")
        print(output)
        x = F.softmax(output,1)
        x = torch.argmax(x, dim=1)
        print(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = make_one_hot(x, C)
        print(x)

        return x


def make_one_hot(labels, C):
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.

        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
        
        C : integer 
        number of classes in labels.

        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
        '''

        one_hot = torch.zeros(size=(labels.size(0), C, labels.size(2), labels.size(3)), dtype=torch.uint8, device=labels.device)

        target = one_hot.scatter_(dim=1, index=labels.to(torch.long), value=1.0)

        # target = Variable(target)

        return target


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

            target = make_one_hot(mask_B_1_H_W, C=4)

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

    # accuracy = accuracy_score(targets, predictions)
    # conf_mat = confusion_matrix(targets, predictions)

    # sensitivity = conf_mat.diagonal()/conf_mat.sum(axis=1)

    # print("Val Accuracy: ", accuracy, "Val Sensitivity (Overall): ", np.mean(sensitivity), "Val loss: ", np.mean(loss))

    print("Val loss: ", np.mean(loss), ", Mean dice: ", np.mean(mean_dice),
    ", Mean dice w/o ch 0: ", np.mean(mean_dice_exc_c_0))

    return np.mean(loss), np.mean(mean_dice), np.mean(mean_dice_exc_c_0)