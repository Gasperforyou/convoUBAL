# Convolutional UBAL
Biologically inspired learning algorithm with convolutional architectures.

Here we present convolutional UBAL (Universal Bidirectional Activation-based Learning), which is a biologically plousable learning algorithm for neural networks. We developed it with convolutional architecture and made some experiments with famous MNIST dataset.

This project is part of **Biologically motivated learning in neural networks with convolutional architectures** Master Thesis.

##Instalation
You need to install python 3.10. There are also several libraries installed. Pytorch 1.13.1, matplotlib, numpy 1.24.1, torchvision 0.14.1

Here we present main structure of python files
## MODEL
 In model folder we have Learning algorithm UBAL for fully connected version and convolutional version written in pytorch library. _UBAL_torch.py_ is model for normal fully-connected UBAL,  _conv_UBAL_torch.py_ is for convolutional layers and conv_autoencoder_UBAL_TORCH.py is specifically for autoencoder.
 
## exp_encoder
In this folder the convolutional autoencoder is implemented. The autoencoder file has an experiment with MNIST dataset.

## exp_mnist
In this folder we implemented fully connected MNIST experiment with perceptual labels, according to perceptual symbol system hypothesis. The main file to run is _mnist_single_net.py_.
### convolutional_mnist

In this folder we implemented convolutional UBAL with first two convolutional layers and the last layer is fully connected one. The main file to run is _convo_mnist_single_net.py_ and _convo_mnist_same_nets.py_. Fist one is only one neural network and the second one are five neueural networks. 

#### percept_convo_UBAL

In this folder is implemented the perceptual convolutional MNIST, with three convolutional layers. The file to run is _convo_mnist_single_net.py_.
 
 
 
 ## License

[MIT](https://choosealicense.com/licenses/mit/)
