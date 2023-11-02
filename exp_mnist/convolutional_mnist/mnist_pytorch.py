#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:07:29 2019

@author: matusmacbookpro
"""
import sys

sys.path.insert(0, '/home/gasper/Dokumenti/faks/kognitivna znanost/izmenjava/semester 1/Projekt/ubal-python')


import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from exp_mnist.mnist_data_provider import MNISTData


#------------define training parameters---------------
log_file_name = 'log_1'

train_data_size = 55000
test_data_size = 10000

minibatch_size = 265
number_of_epochs = 120

hidden_neurons = 1
no_hid_neurons_conv1 = 1
no_hid_neurons_conv2 = 1

learning_rate = 0.0001

def show_intermediate_images(images):
    num_row = 2
    num_col = 5
    plt.figure()
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i, img in enumerate(images.detach()):
        img = img.view(images.size(2), images.size(3))
        img_np = img.cpu().numpy()
        ax = axes[i // num_col, i % num_col]
        ax.imshow(img_np, cmap='gray')
        ax.set_title('Label: {}'.format(i))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(img_np, cmap='gray')
    plt.tight_layout()
    plt.show()

#------------define network------------

#every network in pytorch needs to be defined as a class that inherits componets from
#nn.Module (for more on inheritance see: https://realpython.com/inheritance-composition-python/)
class Net(nn.Module):
    
#    in the init function we initialize layers of our network
    def __init__(self,no_hid_neurons,no_hid_neurons_conv1,no_hid_neurons_conv2):
        
        self.no_hid_neurons = no_hid_neurons
        self.no_hid_neurons_conv1 = no_hid_neurons_conv1
        self.no_hid_neurons_conv2 = no_hid_neurons_conv2
        
#        this command calls __init__ function from nn.Module, THIS ALWAYS NEEDS TO BE DONE!!!
        super(Net, self).__init__()
        
#        define convolutional layer with receptive field size 3x3, stride 2x2 and padding 1,
#        number of neurons is defined by no_hid_neurons_conv1 variable
#        the input size is 1 because we use black and white images (1 channel) as an input
        self.conv1 = nn.Conv2d(1,self.no_hid_neurons_conv1,5,2,0)
        
        self.conv2 = nn.Conv2d(self.no_hid_neurons_conv1,self.no_hid_neurons_conv2,4,1,0)

#        here we define fully connected layer with dimensionality defined by the dimensionality of the previous layer
#        this layer has 10 neurons, one for each category in dataset
        self.fully_connected_out = nn.Linear(9*9,10)
    
#    forward function defines the computation done by the network
    def forward(self,batch, epoch):
        
#        here we define the output of first layer of our network
#        we also need to apply an activation function to the raw output of our layer
        
        activations = torch.relu(self.conv1(batch))
        if epoch == number_of_epochs-1:
            show_intermediate_images(activations[0:10])
        activations = torch.relu(self.conv2(activations))
        if epoch == number_of_epochs-1:
            show_intermediate_images(activations[0:10])
#        here we are reshaping activations array from previous layer so that
#        they are compatible with an input to fully connected layer fully_connected_1
#        we are reshaping activations from size [minibatch_size,self.no_hid_neurons_conv2,7,7] to size: [minibatch_size,7*7*self.no_hid_neurons_conv2]
        activations = activations.view(-1,9*9*self.no_hid_neurons_conv2)
        
#        here we define the output of last layer of our network
        output = torch.relu(self.fully_connected_out(activations))
        
#        return the output
        return output

#---------------load data-------------------
# data = np.load('train_data_2.npz')

data_provider = MNISTData(train_size=train_data_size, validation_size=5000,
                          target_type='one-hot',
                          minibatch_size = minibatch_size,
                          dl_kwargs={'num_workers': 6, 'pin_memory': True})
device = "cuda:0"

# train_images = data['train_images']
# test_images = data['test_images']
# train_labels = data['train_labels']
# test_labels = data['test_labels']
#
# train_images = train_images[0:train_data_size,:,:,:].astype(np.float32)
# test_images = test_images[0:test_data_size,:,:,:].astype(np.float32)
# train_labels = train_labels[0:train_data_size,:].astype(np.float32)
# test_labels = test_labels[0:test_data_size,:].astype(np.float32)
#
# #--------------normalize and reshape data---------------------
# train_images = train_images/255.
# test_images = test_images/255.
#
# #change the shape of train images from [no_images,28,28,3] to [no_images,3,28,28]
# # we do this because pytorch accepts inputs to convolutional layer in this shape: [batch_size,number_of_channels,resolution_x,resolution_y]
# train_images = np.rollaxis(train_images,3,1)
# #we do the same for test images
# test_images = np.rollaxis(test_images,3,1)


#if we want work with variables in pytorch we need to convert these variables to a format that pytorch understands
# torch.from_numpy converts numpy array to an tensor format that can be manipulated by pytorch
# Variable() creates an pytorch variable whtic difereich enables the automantiation and gradient computation
test_images_pytorch = Variable(data_provider.test_images.view(-1, 1, data_provider.image_side, data_provider.image_side).to(device))

#---------------initialize key network components-----------------------

# initialize network class
network = Net(hidden_neurons,no_hid_neurons_conv1,no_hid_neurons_conv2).to(device)

# define the loss function, calculates the difference between target output and network output
loss_function = nn.MSELoss()

# define the optimizer, optimizer calculates the new values for network weights
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

#-------------------main training loop-----------------------
for ep in range(number_of_epochs):
    
    #rand = np.random.permutation(train_data_size)

#--------inner training loop (batch generation and network training)------------

    for i, (images, labels) in enumerate(data_provider.train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
#        zeroes out the value of gradient from previous weight change (THIS NEEDS TO BE DONE AFTER EVERY NETWORK UPDATE)
        optimizer.zero_grad()
        
        # selected_images = train_images[rand[i:i+minibatch_size],:,:,:]
        # selected_labels = train_labels[rand[i:i+minibatch_size],:]
        #
#        create variables for this minibatch
        batch_images = Variable(images).to(device)
        batch_labels = Variable(labels).to(device)
        
#        calculate the output of the network for this minibatch
        net_output = network.forward(batch_images, False)
        
#        calculate the loss using network output and labels
        loss = loss_function(net_output,batch_labels)
        
#        calculate the gradients for this minibatch
        loss.backward()
        
#        change the weights of the network
        optimizer.step()
    
#----------------test the accuracy of the network------------------
        
#   calculate the output of the network for test images
#   with torch.no_grad() enables to calculates the output of the network without gradient calculation, which is much more computationally efficient 
    with torch.no_grad():
        
        class_output = network.forward(test_images_pytorch, ep)

# convert the output of neural network to numpy format    
    class_output_numpy = class_output.cpu().numpy()
    
    arg_out = np.argmax(class_output_numpy,axis = 1)
    arg_class = np.argmax(data_provider.test_labels.numpy(),axis = 1)
    
    score = np.sum(arg_out == arg_class)
    print('Classification accuracy at epoch: ' + str(ep) + ' : ' + str((score/test_data_size)*100) + '%')

