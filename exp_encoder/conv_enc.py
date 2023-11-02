import sys
sys.path.insert(1, '/home/gasper/Dokumenti/faks/kognitivna znanost/izmenjava/semester 1/Projekt/ubal-python')

from model.conv_autoencoder_UBAL_torch import ConvAutoencoderUBAL, UBALoptim

import torch

torch.set_printoptions(threshold = 100000)


num_epochs = 30
max_batches = 20
batch_size = 128
learning_rate = 0.01
use_gpu = True

# parameters for ubal
betas = [0.5, 0.5, 0.5, 0.5, 0.5]
gammasF = [float("nan"), 1.0, 1.0, 1.0, 1.0]
gammasB = [1.0, 1.0, 1.0, 1.0, float("nan")]

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# autoencoder initialization
autoencUBAL = ConvAutoencoderUBAL()

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
autoencUBAL = autoencUBAL.to(device)

num_params = sum(p.numel() for p in autoencUBAL.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)


optimizer = UBALoptim(autoencUBAL.parameters(), betas, gammasF, gammasB, device, lr=learning_rate)

# set to training mode
autoencUBAL.train()

def to_img_train(x):
    x = 0.5 * (x + 1)
    return x
print('Training ...')
torch.set_grad_enabled(False)
for epoch in range(num_epochs):
    num_batches = 0

    for image_batch, _ in train_dataloader:
        image_batch = to_img_train(image_batch).to(device)

        # autoencoder reconstruction
        act_FP, act_BP, act_FE, act_BE = autoencUBAL(image_batch, image_batch)
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step(act_FP, act_BP, act_FE, act_BE)


        num_batches += 1
        if (num_batches >= max_batches):
            break

    print('Epoch [%d / %d]' % (epoch + 1, num_epochs))


import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils

autoencUBAL.eval()

# This function takes as an input the images to reconstruct
# and the name of the model with which the reconstructions
# are performed

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x
def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images, model):
        images = to_img_train(images).to(device)
        images, back, _, _ = model(images, images)
        images = images[4].cpu()
        print(images[0])
        #images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

        back = back[0].cpu()
        np_imagegrid = torchvision.utils.make_grid(back[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()


images, labels = next(iter(test_dataloader))

# Reconstruct and visualise the images using the autoencoder
print('Autoencoder reconstruction:')
visualise_output(images, autoencUBAL)

# First visualise the original images
print('Original images')
show_image(torchvision.utils.make_grid(images[1:50],10,5))
plt.show()
