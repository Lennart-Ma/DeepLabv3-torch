import torch
from PIL import Image
from torchvision import transforms
import ntpath



def from_channel_to_label_tensor(channel_tensor):
        """
        Converts a channel tensor (e.g. one_hot tensor or tensor with probabilities along dim 1 of shape (N x C x H x W) = (batchsize, n_channels(n_classes), height, width)
        to a int label tensor of shape (batchsize, 1, height, width) with each pixel of height,width corresponds to a class label (0, .., C-1)

        Parameters
        ----------
        one_hot_mask : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
            
        C : integer 
        number of classes in labels.

        Returns
        -------

        labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing the label (class).
        """

        labels = torch.argmax(channel_tensor, dim=1)
        labels = labels.reshape(labels.shape[0], 1, labels.shape[1], labels.shape[2])

        return labels


def make_one_hot(labels, C):
        '''
        Converts an integer label tensor (N x 1 x H x W) to a one-hot tensor of shape (N x C x H x W).

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




def create_transparent_overlay(image, seg_map):
        """
        Takes as input an image (dtype:PIL grayscale) and the corresponding predicted segmentation mask (dtype:PIL grayscale)
        and creates a new image with transparent overlay of the segmentation mask over the image
        Class 0 --> full transparent
        Class 1 --> green transparent
        Class 2 --> red transparent
        Class 3 --> blue transparent
        """

        rgbmask = Image.new("RGBA", seg_map.size)
        rgbmask.paste(seg_map)

        rgbimage = Image.new("RGBA", image.size)
        rgbimage.paste(image)

        pixels = rgbmask.load()

        for i in range(rgbmask.size[0]): # for every pixel:
                for j in range(rgbmask.size[1]):
                        if pixels[i,j] == (0, 0, 0, 255):
                                # change to full transparent if class 0
                                pixels[i,j] = (0, 0 , 0, 0)
                        if pixels[i,j] == (1, 1, 1, 255):
                                # change to green if class 1
                                pixels[i,j] = (0, 128 , 0, 50)
                        if pixels[i,j] == (2, 2, 2, 255):
                                # change to red if class 2
                                pixels[i,j] = (255, 0 , 0, 50)
                        if pixels[i,j] == (3, 3, 3, 255):
                                # change to blue if class 3
                                pixels[i,j] = (0, 0, 255, 50)

        foreground = rgbmask


#### FOREGROUND (transparent seg map) works but the background somehow does not work!!! Try out how overlay works!

        background = rgbimage

        transparent_overlay = Image.alpha_composite(background, foreground)

        return transparent_overlay
        

def draw_seg_map_from_single_image(image_file, device, model):

        """
        Predicts the segmentation map of a given image_file with a given model and saves it under output_path "test_mask.png"


        Parameters
                ----------
                image_file : String
                path to the image (.png) that is used for segmentation map prediction
                
                device : String
                On which device data and model will be 
                
                model: .bin
                A trained model saved as *.bin which will be loaded to predict the seg map to the image_file

                output_path: String
                path to the folder where the predicted segmentation map will be saved as test_mask.png 

                Returns
                -------

                PIL_image : PIL image

        """

        image_PIL = Image.open(image_file).convert("L")

        normalize = transforms.ToTensor()

        image = normalize(image_PIL)

        image = torch.tensor([image.numpy(), image.numpy()])

        image.to(device)

        if device == "cuda":
                image = image.type(torch.cuda.FloatTensor)

        print(image.shape)

        output = model(image)

        print(output['out'].shape)

        label_images = from_channel_to_label_tensor(output['out'])

        print(label_images[0].shape)

        # basename = ntpath.basename(image_file)
        # k = basename.rfind(".")
        # basename = basename[:k]

        seg_mask = transforms.ToPILImage()(label_images[0].type(torch.uint8))
        
        return image_PIL, seg_mask



def create_basename_from_input_path(input_path):

        basename = ntpath.basename(input_path)
        k = basename.rfind(".")
        basename = basename[:k]

        return basename