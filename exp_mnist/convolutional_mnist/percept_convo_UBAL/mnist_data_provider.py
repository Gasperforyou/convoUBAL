import os
from torchvision import datasets, transforms
import torch
import ast


class OneHotTransform:
    def __call__(self, x):
        transformed = torch.zeros(10)
        transformed[x - 1] = 1
        return transformed

class Perceptual:
    def __call__(self, x):
        with open("exp_mnist/convolutional_mnist/percept_convo_UBAL/perceptual_labels.py") as f:
            percep_labels = f.read()
            percep_labels = ast.literal_eval(percep_labels)
            label = torch.FloatTensor(percep_labels[str(x)])
        return label


class MNISTData:
    image_side = 28
    image_size = 784
    no_classes = 10
    no_output_neurons = 15

    def __init__(self, train_size=55000, validation_size=5000,
                 minibatch_size=256, random_seed=0,
                 target_type='one_hot', dl_kwargs={}):
        self.train_size = train_size
        self.validation_size = validation_size
        self.minibatch_size = minibatch_size
        if random_seed > 0:
            self.generator = torch.Generator().manual_seed(random_seed)
        else:
            self.generator = torch.Generator()
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        if target_type == 'perceptual':
            label_transform = Perceptual()
        else:
            label_transform =  OneHotTransform()
        trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform,
                                  target_transform=label_transform)
        trainset = torch.utils.data.Subset(trainset, range(train_size + validation_size))
        train_subset, validation_subset = torch.utils.data.random_split(
            trainset, [train_size, validation_size], generator=self.generator)
        self.train_loader = torch.utils.data.DataLoader(train_subset, batch_size=minibatch_size, shuffle=True,
                                                        **dl_kwargs)

        validation_images = []
        validation_labels = []
        for index, i in enumerate(validation_subset.indices):
            image, target = validation_subset.dataset.__getitem__(i)
            validation_images.append(image)
            validation_labels.append(target)
        self.validation_images = torch.stack(validation_images)
        self.validation_labels = torch.stack(validation_labels)

        testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform,
                                 target_transform=label_transform)
        test_images_list = []
        test_labels_list = []
        for i, (image, target) in enumerate(testset):
            test_images_list.append(image)
            test_labels_list.append(target)
        self.test_images = torch.stack(test_images_list)
        self.test_labels = torch.stack(test_labels_list)

    # def random_data_train(self, max_count=1):

    def mnist_labels(self, add_noise=False):
        mnist_labels = torch.zeros(10, 10)
        for i in range(10):
            mnist_labels[i][i] = 1
        if add_noise:
            mnist_labels = mnist_labels + torch.randn(mnist_labels.size()) * self.noise_variance + self.noise_mean
        return mnist_labels

    def percep_mnist_labels(self):
        with open("exp_mnist/convolutional_mnist/percept_convo_UBAL/perceptual_labels.py") as f:
            percep_labels = f.read()
            percep_labels = ast.literal_eval(percep_labels)

        tuple = ()
        for i in percep_labels:
            tuple += (torch.FloatTensor(percep_labels[str(i)]), )
        labels = torch.stack(tuple, 0)
        return labels