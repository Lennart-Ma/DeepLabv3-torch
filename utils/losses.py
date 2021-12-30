import torch
from torch import nn



class Generalized_Dice_Loss(nn.Module):
    """
    Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    #def __init__(self, PARAMS, epsilon=1e-5):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        #self.ignore_class_index = PARAMS['DICE_LOSS_IGNORE_CLASS_INDEX']
        self.ignore_class_index = 3
        self.apply_softmax = False
        self.apply_sigmoid = False

        #if PARAMS['FINAL_ACTIVATION_SEG'] == 'softmax':
        self.apply_softmax = True
        #if PARAMS['FINAL_ACTIVATION_SEG'] == 'sigmoid':
        #    self.apply_sigmoid = True

        #if 'Shape_Prior' in PARAMS['MODEL']:
        #    self.apply_softmax = False

        assert not (self.apply_softmax and self.apply_sigmoid), 'Softmax and sigmoid cannot both be applied.'

        self.logger = []
        self.counter = 1

    def __str__(self):
        string = 'GDL'
        string += 'no' + str(self.ignore_class_index) if self.ignore_class_index is not None else ''
        return string

    def forward(self, masks_pred, masks_true):
        # [n, c, h, w]
        # get probabilities from logits
        if self.apply_softmax:
            masks_pred = torch.softmax(masks_pred, dim=1)
        if self.apply_sigmoid:
            masks_pred = torch.sigmoid(masks_pred)
        
        batch_size, num_classes, _, _ = masks_pred.shape

        assert masks_pred.size() == masks_true.size(), "'masks_pred' and 'masks_true' must have the same shape but have shape {0} and {1}".format(masks_pred.shape, masks_true.shape)

        masks_true = masks_true.float()

        nominators = torch.zeros(size=(batch_size, num_classes))
        denominators = torch.zeros(size=(batch_size, num_classes))
        for j in range(num_classes):
            if j == self.ignore_class_index:
                print("Ignored Class Index")
                continue
            class_weight = 1.0 / masks_true[:, j, :, :].sum(dim=(-2, -1)).pow(2).clamp(min=self.epsilon)
            nominators[:, j] = (masks_pred[:, j, :, :] * masks_true[:, j, :, :]).sum(dim=(-2, -1)) * class_weight
            denominators[:, j] = (masks_pred[:, j, :, :] + masks_true[:, j, :, :]).sum(dim=(-2, -1)) * class_weight
        
        gen_dice_loss = (1.0 - 2.0 * nominators.sum(-1) / denominators.sum(-1)).mean()

        """
        if self.counter % 50 == 0 or self.counter % 200 == 1:
            _ = verification_plot(masks_pred.detach().cpu().numpy()[:, 0, ...])
            _ = verification_plot(masks_pred.detach().cpu().numpy()[:, 1, ...])
            _ = verification_plot(masks_true.detach().cpu().numpy()[:, 0, ...])
            _ = verification_plot(masks_true.detach().cpu().numpy()[:, 1, ...])
            print('DICE LOSS', gen_dice_loss)
            plt.show()
        
        self.counter += 1
        """
        self.logger.append(gen_dice_loss.item())
        return gen_dice_loss
