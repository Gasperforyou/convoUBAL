import sys
sys.path.insert(1, '/home/gasper/Dokumenti/faks/kognitivna znanost/izmenjava/semester 1/Projekt/ubal-python')
import pickle
import time
import torch
torch.cuda.empty_cache()
import numpy as np
from exp_mnist.convolutional_mnist.convo_net import ConvNet

from exp_mnist.mnist_data_provider import MNISTData
import matplotlib.pyplot as plt
from torchvision import transforms

class UBALSim:
    def __init__(self, settings_conv, settings_trans, learning_rate, init_w_mean, init_w_var,
               betas, gammasF, gammasB, epochs, act_fun_F, act_fun_B, data_provider, device, results_path,
                 target_noise_m=0.0, target_noise_variance=0.00005, log_freq=10):
        self.act_fun_F = act_fun_F
        self.act_fun_B = act_fun_B
        self.settings_conv = settings_conv
        self.settings_trans = settings_trans
        self.learning_rate = learning_rate
        self.init_w_mean = init_w_mean
        self.init_w_var = init_w_var
        self.betas = betas
        self.gammasF = gammasF
        self.gammasB = gammasB
        self.device = device
        self.train_epochs = epochs
        self.data_provider = data_provider
        self.device = device
        self.target_noise_m = target_noise_m
        self.target_noise_variance = target_noise_variance
        self.results_path = results_path
        self.logging_frequency = log_freq

    def add_noise_gauss(self, input, mean, var):
        return input + torch.randn(input.size()) * var + mean

    def mean_squared_error(self, desired, estimate):
        return ((desired - estimate)**2).mean()

    def train_test_one_net(self, save_net=False, save_backward_images=False, test_each_ep=False, noisy_targets=False):
        output_performance = {"acc_f_train": [], "acc_f_validation": [], "acc_f_test" : [],
                              "mse_b_train": [], "mse_b_validation": [], "mse_b_test" : [],
                              "acc_backward_images": []}
        net = ConvNet(self.settings_conv, self.settings_trans, self.act_fun_F, self.act_fun_B,  self.learning_rate, self.betas, self.gammasF, self.gammasB, self.device)
        start_time = time.time()
        for epoch in range(self.train_epochs):
            acc_train = 0
            mse_b_train = 0
            net.train()
            for i, (images, labels) in enumerate(self.data_provider.train_loader):
                images = images.to(self.device)
                if noisy_targets:
                    labels = self.add_noise_gauss(labels, self.target_noise_m, self.target_noise_variance)
                labels = labels.to(self.device)
                # propragate activations through network
                act_fp, act_fe, act_bp, act_be, act_fp_normal, act_fe_normal, act_bp_normal, act_be_normal = net.activation(images, labels)
                # learn
                net.learn(act_fp, act_fe, act_bp, act_be, act_fp_normal, act_fe_normal, act_bp_normal, act_be_normal)
                train_output_winners = torch.argmax(act_fp_normal[-1], axis=1)
                train_target_winners = torch.argmax(labels, axis=1)
                acc_train += torch.sum(train_output_winners == train_target_winners)
                mse_b_train += self.mean_squared_error(images, transforms.Pad((0, 0, 1, 1))(act_bp[0]))
            net.eval()
            acc_train = acc_train / self.data_provider.train_size
            mse_b_train = mse_b_train / self.data_provider.train_size
            output_performance["acc_f_train"].append(acc_train.item())
            output_performance["mse_b_train"].append(mse_b_train.item())

            validation_images = self.data_provider.validation_images.view(-1, 1 , self.data_provider.image_side, self.data_provider.image_side).to(self.device)
            validation_output = net.activation_FP_last(validation_images)
            val_output_winners = torch.argmax(validation_output, axis=1)
            val_target_winners = torch.argmax(self.data_provider.validation_labels.to(self.device), axis=1)
            acc_valid = torch.sum(val_output_winners == val_target_winners) / self.data_provider.validation_size
            output_performance["acc_f_validation"].append(acc_valid.item())

            mnist_labels = torch.zeros(10, 10)
            for i in range(10):
                mnist_labels[i][i] = 1
            mnist_labels = mnist_labels.to(self.device)
            act_bp_last = net.activation_BP_last(mnist_labels)
            act_fp_last = net.activation_FP_last(act_bp_last)
            acc_backward_image = torch.sum(torch.argmax(act_fp_last, axis=1) == torch.argmax(mnist_labels, axis=1)) / len(mnist_labels)
            output_performance["acc_backward_images"].append(acc_backward_image.item())

            # if acc_backward_image.item() == 1.0:
            #     filename = "lr{}_beta2f{}_{}epcs.{}.png".format(
            #         self.learning_rate, self.betas[1], self.train_epochs, int(start_time))
            #     file_path = self.results_path + "images/" + filename
            #     self.save_backward_images(net, file_path)
            #     filename = "lr{}_{}epcs.{}.pickle".format(
            #         self.learning_rate, self.train_epochs, int(start_time))
            #     if noisy_targets:
            #         filename = "tnoise{}_".format(self.target_noise_variance) + filename
            #     net.save_weights(self.results_path + "weights/" + filename)

            if test_each_ep:
                test_images = self.data_provider.test_images.view(-1, 1, self.data_provider.image_side, self.data_provider.image_side).to(self.device)
                test_labels = self.data_provider.test_labels.to(self.device)
                test_output = net.activation_FP_last(test_images)
                test_output_winners = torch.argmax(test_output, axis=1)
                test_target_winners = torch.argmax(test_labels, axis=1)
                acc_test = torch.sum(test_output_winners == test_target_winners) / len(test_images)
                output_performance["acc_f_test"].append(acc_test.item())

            end_time = time.time()
            if ((epoch+1) % self.logging_frequency == 0):
                if test_each_ep:
                    print("Epoch {}. ACC-Train: {:.3f}% ACC-Val: {:.3f}% ACC-Back: {:.3f}% Time: {:.1f} seconds".format(
                    epoch+1, acc_train*100, acc_valid*100, acc_test*100, acc_backward_image*100, end_time - start_time))
                else:
                    print("Epoch {}. ACC-Train: {:.3f}% ACC-Val: {:.3f}% ACC-Back: {:.3f}% MSE-Train: {:.3f}%. Time: {:.1f} seconds".format(
                    epoch+1, acc_train*100, acc_valid*100, acc_backward_image*100, mse_b_train*100, end_time - start_time))
            start_time = end_time

        test_images = self.data_provider.test_images.view(-1, 1, self.data_provider.image_side, self.data_provider.image_side).to(self.device)
        test_labels = self.data_provider.test_labels.to(self.device)
        act_fp, _ = net.forward(test_images)
        self.show_intermediate_images(act_fp[1][0:10])
        self.show_intermediate_images(act_fp[2][0:10])
        test_output = net.activation_FP_last(test_images)
        test_output_winners = torch.argmax(test_output, axis=1)
        test_target_winners = torch.argmax(test_labels, axis=1)
        acc_test = torch.sum(test_output_winners == test_target_winners) / len(test_images)
        output_performance["acc_f_test"].append(acc_test.item())
        print("Testing Accuracy: {:.3f}%".format(acc_test*100))

        mnist_labels = torch.zeros(10, 10)
        for i in range(10):
            mnist_labels[i][i] = 1
        mnist_labels = mnist_labels.to(self.device)
        act_bp_last = net.activation_BP_last(mnist_labels)
        act_fp_last = net.activation_FP_last(act_bp_last)
        acc_backward_image = torch.sum(torch.argmax(act_fp_last, axis=1) == torch.argmax(mnist_labels, axis=1)) / len(
            mnist_labels)
        print("Backward Accuracy: {:.3f}%".format(acc_backward_image * 100))




        if acc_backward_image == 1:
            filename = "hids{}_lr{}_beta2f{}_epc{}.{}.png".format(
                self.hidden_neurons, self.learning_rate, self.betas[1], epoch+1, int(start_time))
            file_path = self.results_path + "images/" + filename
            self.save_backward_images(net, file_path)

        # if save_net:
        #     filename = "hids{}_lr{}_beta3f{}_{}epcs.{}.pickle".format(
        #         self.hidden_neurons, self.learning_rate, self.betas[net.d - 1], self.train_epochs, int(start_time))
        #     if noisy_targets:
        #         filename = "tnoise{}_".format(self.target_noise_variance) + filename
        #     net.save_weights(self.results_path + "weights/" + filename)

        if save_backward_images:
            filename = "lr{}_beta2f{}_{}epcs.{}.png".format(
                self.learning_rate, self.betas[1], self.train_epochs, int(start_time))
            file_path = self.results_path + "images/" + filename
            self.save_backward_images(net, file_path)

        return output_performance

    def save_backward_images(self, net, file_path, num_row=2, num_col=5):
        labels = self.data_provider.mnist_labels()
        act_bp = net.activation_BP_last(labels.to(self.device))
        plt.figure()
        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
        for i, img in enumerate(act_bp):
            img = img.view(27, 27)
            img_np = img.cpu().detach().numpy()
            ax = axes[i // num_col, i % num_col]
            ax.imshow(img_np, cmap='gray')
            ax.set_title('Label: {}'.format(torch.argmax(labels[i]).item() + 1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(img_np, cmap='gray')
        plt.tight_layout()
        plt.savefig(file_path)
        print("Saved image to {}.".format(file_path))

    def show_intermediate_images(self, images):
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