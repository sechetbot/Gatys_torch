
'''
An implementation of "A Neural Algorithm of Artistic Style", by Leon Gatys, Alexander Ecker and Matthias Bethge
The paper can be found at: https://arxiv.org/pdf/1508.06576.pdf
This tutorial was followed: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
'''

import time
import matplotlib.pyplot as plt
import PIL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

def load_image(name, imsize):
    image = PIL.Image.open('Images/' + name)
    image = transforms.Resize([imsize,imsize])(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0) #give a batch dimension of 1 to fit the networks input dimensions
    image = image.to(device)
    return image

def show_images(image_array):
    plt.figure()
    
    for counter, image in enumerate(image_array):
        image = image.squeeze(0).cpu() #remove batch dimension
        image = transforms.ToPILImage()(image)
        plt.subplot(int('1' + str(len(image_array)) + str(counter+1)))
        plt.imshow(image)

    plt.show()


#define content and style losses
class Contentloss(nn.Module):
    def __init__(self, target):
        super(Contentloss, self).__init__() #pytorch syntax for writing modules, look at this
        self.target = target.detach() #target is not a variable so if left attached will raise error when forward method called
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


def gram(feature_maps): #refer to https://www.youtube.com/watch?v=DEK-W5cxG-g for why it is calculated like this
    n_batch, n_maps, h, w = feature_maps.size()
    flattened_maps = feature_maps.view(n_batch*n_maps, h*w)
    #flattened_maps is a matrix of n flattened feature maps(filter results) stacked up

    matrix = torch.mm(flattened_maps, flattened_maps.t()) #this step is the most RAM intensive I believe

    #normalisation as used by Gatys et al. (this is crucial and often forgotten)
    return matrix/(n_batch*n_maps*h*w)


class Styleloss(nn.Module):
    def __init__(self, target):
        super(Styleloss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = nn.functional.mse_loss(gram(input), gram(self.target))
        return input


class Normalise(nn.Module):
    def __init__(self, mean, std):
        super(Normalise, self).__init__()
        self.mean = mean.unsqueeze(1).unsqueeze(2)
        self.std = mean.unsqueeze(1).unsqueeze(2)
        #change the mean and std vectors to be in form [channels, height, width]
    def forward(self, img):
        return (img - self.mean)/self.std


#we now create a new sequential model with the loss modules placed after the correct layers
default_content_layers = ['Conv_4']
default_style_layers = ['Conv_1', 'Conv_2', 'Conv_3', 'Conv_4', 'Conv_5']

def get_model_and_losses(cnn, norm_mean, norm_std, content_img, style_img, 
                    content_layers=default_content_layers, style_layers=default_style_layers):
    
    content_losses = []
    style_losses = []

    normalise_instance = Normalise(norm_mean, norm_std).to(device)

    #creating the new model
    model = nn.Sequential(normalise_instance)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i+=1
            name = 'Conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'ReLU_{}'.format(i)

            layer = nn.ReLU(inplace=False) #the inplace ones do not work with the loss modules created
        elif isinstance(layer, nn.MaxPool2d):
            name = 'AvgPool_{}'.format(i)

            layer = nn.AvgPool2d(2, stride=2, padding=0, ceil_mode=False) #we replace MaxPool layers with AvgPool layers as done by Gatys et al.
        elif isinstance(layer, BatchNorm2d):
            name = 'BatchNorm_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
    
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            contentloss_instance = Contentloss(target)
            model.add_module('contentloss_{}'.format(i), contentloss_instance)
            content_losses.append(contentloss_instance)

        if name in style_layers:
            target = model(style_img).detach()
            styleloss_instance = Styleloss(target)
            model.add_module('styleloss_{}'.format(i), styleloss_instance)
            style_losses.append(styleloss_instance)


    #trim off the layers after the last content or style loss instance
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], Contentloss) or isinstance(model[i], Styleloss):
            break

    model = model[:(i+1)]

    return model, content_losses, style_losses


#we define an L-BFGS optimiser that will do gradient descent on our image
def get_optimiser(image):
    optimiser = torch.optim.LBFGS([image.requires_grad_()]) #ensure the image requres a gradient
    return optimiser


#function that does the transfer
def transfer_style(content_img, style_img, input_img, cnn, norm_mean, norm_std, n_iter=1000,
                    style_weight=1e7, content_weight=1):
    
    start_time = time.time()
    batchtimes = [start_time]

    print('Getting model and optimiser')
    model, content_losses, style_losses = get_model_and_losses(cnn, norm_mean, norm_std, content_img, style_img)
    optimiser = get_optimiser(input_img)

    print('Optimising')
    i = [0]
    while i[0] <= n_iter:

        def closure(): #the closure function takes an input and computes losses from it
            input_img.clamp(0,1)

            optimiser.zero_grad() #we need to zero the gradient from the last cycle of the closure fn
            model(input_img)
            style_loss = 0
            content_loss = 0

            for loss_instance in style_losses:
                style_loss += loss_instance.loss

            for loss_instance in content_losses:
                content_loss += loss_instance.loss

            style_loss *= style_weight
            content_loss *= content_weight

            loss = style_loss + content_loss
            loss.backward() #runs backwards method of our loss modules to find the gradient

            i[0] += 1

            if i[0] % 50 == 0:
                batchtimes.append(time.time())
                batch_time = batchtimes[int(i[0]/50)] - batchtimes[int(i[0]/50)-1]
                print("run {}:".format(i[0]))
                print('Style Loss : {}'.format(style_loss))
                print('Content Loss: {}'.format(content_loss))
                print('Batch time: {}'.format(batch_time))
                print()

            return loss
        
        optimiser.step(closure) #carries out one optimisation step and recarries out closure

    input_img = input_img.clamp(0,1)

    print('Total time taken: {}'.format(time.time() - start_time))

    return input_img



######################

torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('The pytorch device being used is: {}'.format(device))

imsize = 600 #change later, remake gram matrix calculation
content = load_image('Blue_and_yellow_macaw.jpg', imsize)
style = load_image('Figure_dans_un_Fauteuil.jpg', imsize)
white_noise = torch.randn([1, 3, imsize, imsize], device=device)

# show_images([turtle, scream, white_noise])



vgg19 = models.vgg19(pretrained=True).features.to(device).eval()
#this pytorch pretrained model is in 2 sequential parts, features and then classifier
#classifier is the fully connected part, which we do not need, so we just import features
#eval just means that the weights cannot be changed, we are not training


norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
#the vgg19 network was trained on images normalised by this mean and std


output = transfer_style(content, style, white_noise, vgg19, norm_mean, norm_std, n_iter=1000)

output = output.detach().squeeze(0).cpu() #cannot change a cuda tensor to numpy, so first change it to cpu
output = transforms.ToPILImage()(output)
plt.imshow(output)
plt.show()






