import pickle
import time
import torch
import numpy as np
from model.conv_UBAL_torch import ConvUBAL, UBALoptim
from exp_mnist.mnist_data_provider import MNISTData
import matplotlib.pyplot as plt

import torchvision.transforms as transforms


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
        net = ConvUBAL(self.settings_conv, self.settings_trans, self.act_fun_F, self.act_fun_B, self.device)
        optimizer = UBALoptim(net.parameters(), self.betas, self.gammasF, self.gammasB, self.device,
                                   self.settings_conv, self.settings_trans, lr=self.learning_rate)

        transform = transforms.Pad((0, 0, 1, 1))
        start_time = time.time()
        net.train()
        for epoch in range(self.train_epochs):
            acc_train = 0
            mse_b_train = 0
            t = torch.Tensor([0.5]).to(self.device)  # threshold
            for i, (images, labels) in enumerate(self.data_provider.train_loader):
                images = images.to(self.device)
                if noisy_targets:
                    labels = self.add_noise_gauss(labels, self.target_noise_m, self.target_noise_variance)
                labels = labels.view(labels.size()[0], 1, labels.size()[1], labels.size()[2])
                labels = labels.to(self.device)
                act_fp, act_fe = net.forward(images)
                act_bp, act_be = net.backward(labels)
                optimizer.step(act_fp, act_fe, act_bp, act_be)
                act_fp = self.normalize(act_fp[-1])
                act_bp = self.normalize(act_bp[0])
                if epoch == 109:
                    train_output_winners = (act_fp >= t).float()
                    train_target_winners = (labels >= t).float()
                    train_output_winners = self.percep_to_one_hot(train_output_winners)
                    train_target_winners = self.percep_to_one_hot(train_target_winners)
                    acc_train += torch.sum(torch.all(train_output_winners.eq(train_target_winners), axis = 1))
                mse_b_train += self.mean_squared_error(images, act_bp)
            acc_train = acc_train / self.data_provider.train_size
            mse_b_train = mse_b_train / self.data_provider.train_size
            output_performance["acc_f_train"].append(acc_train)
            output_performance["mse_b_train"].append(mse_b_train.item())

            validation_output, _ = net.forward(self.data_provider.validation_images.to(self.device))
            validation_output = self.normalize(validation_output[-1])
            val_output_winners = (validation_output > t).float()
            val_target_winners = (self.data_provider.validation_labels.to(self.device) > t).float()
            val_output_winners = self.percep_to_one_hot(val_output_winners)
            val_target_winners = self.percep_to_one_hot(val_target_winners)
            acc_valid = torch.sum(torch.all(val_output_winners.eq(val_target_winners), axis=1))/ self.data_provider.validation_size
            output_performance["acc_f_validation"].append(acc_valid.item())

            mnist_labels = self.data_provider.percep_mnist_labels()
            mnist_labels = mnist_labels.view(mnist_labels.size()[0], 1, mnist_labels.size()[1], mnist_labels.size()[2])
            mnist_labels = mnist_labels.to(self.device)
            act_bp_last, _ = net.backward(mnist_labels)
            act_fp_last, _ = net.forward(act_bp_last[0])
            act_fp_last = self.normalize(act_fp_last[-1])
            act_fp_last = (act_fp_last >= t).float()
            mnist_labels = (mnist_labels >= t).float()
            act_fp_last = self.percep_to_one_hot(act_fp_last)
            mnist_labels = self.percep_to_one_hot(mnist_labels)
            acc_backward_image = torch.sum(torch.all(act_fp_last.eq(mnist_labels), axis=1))/mnist_labels.size(0)
            output_performance["acc_backward_images"].append(acc_backward_image.item())
            # if acc_backward_image.item() == 1.0:
            #     filename = "hids{}_lr{}_beta2f{}_{}epcs.{}.png".format(
            #         self.hidden_neurons, self.learning_rate, self.betas[1], self.train_epochs, int(start_time))
            #     file_path = self.results_path + "images/" + filename
            #     self.save_backward_images(net, file_path)
            #     filename = "hids{}_lr{}_beta3f{}_{}epcs.{}.pickle".format(
            #         self.hidden_neurons, self.learning_rate, self.betas[net.d - 1], self.train_epochs, int(start_time))
            #     if noisy_targets:
            #         filename = "tnoise{}_".format(self.target_noise_variance) + filename
            #     net.save_weights(self.results_path + "weights/" + filename)

            # test each epoch
            if test_each_ep:
                test_images = self.data_provider.test_images.to(self.device)
                test_labels = self.data_provider.test_labels.to(self.device)
                test_output = net.forward(test_images)
                test_output = self.normalize(test_output[-1])
                test_output_winners = (test_output >= t).float()
                test_target_winners = (test_labels >= t).float()
                percent_correct = torch.sum(test_output_winners.eq(test_target_winners), axis=1) / test_labels.size(1)
                acc_test = torch.sum(percent_correct >= 1) / test_labels.size(0)
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

        net.eval()
        # Testing accuracy
        test_images = self.data_provider.test_images.to(self.device)
        test_labels = self.data_provider.test_labels.to(self.device)
        test_output, test_backward = net.forward(test_images)
        self.show_intermediate_images(test_output[0][0:10])
        self.show_intermediate_images(test_output[1][0:10])
        self.show_intermediate_images(test_output[2][0:10])
        self.show_intermediate_images(test_output[3][0:10])
        test_output = self.normalize(test_output[-1])
        test_output_winners = (test_output >= t).float()
        test_target_winners = (test_labels >= t).float()
        test_output_winners = self.percep_to_one_hot(test_output_winners)
        test_target_winners = self.percep_to_one_hot(test_target_winners)
        acc_test = torch.sum(torch.all(test_output_winners.eq(test_target_winners), axis=1)) / test_labels.size(0)
        output_performance["acc_f_test"].append(acc_test.item())
        print("Testing Accuracy: {:.3f}%".format(acc_test*100))

        # Backward accuracy
        mnist_labels = self.data_provider.percep_mnist_labels().to(self.device)
        mnist_labels = mnist_labels.view(mnist_labels.size()[0], 1, mnist_labels.size()[1], mnist_labels.size()[2])
        act_bp_last, _ = net.backward(mnist_labels)
        act_fp_last, _ = net.forward(act_bp_last[0])
        act_fp_last = self.normalize(act_fp_last[-1])
        act_fp_last = (act_fp_last >= t).float()
        mnist_labels = (mnist_labels >= t).float()
        act_fp_last = self.percep_to_one_hot(act_fp_last)
        mnist_labels = self.percep_to_one_hot(mnist_labels)
        acc_test = torch.sum(torch.all(act_fp_last.eq(mnist_labels), axis=1)) / mnist_labels.size(0)
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
        labels = self.data_provider.percep_mnist_labels()
        labels = labels.view(labels.size()[0], 1, labels.size()[1], labels.size()[2])
        act_bp , _ = net.backward(labels.to(self.device))
        act_bp = self.normalize(act_bp[0])
        plt.figure()
        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
        for i, img in enumerate(act_bp.detach()):
            img = img.view(self.data_provider.image_side, self.data_provider.image_side)
            img_np = img.cpu().numpy()
            ax = axes[i // num_col, i % num_col]
            ax.imshow(img_np, cmap='gray')
            ax.set_title('Label: {}'.format(i))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(img_np, cmap='gray')
        plt.tight_layout()
        plt.savefig(file_path)
        print("Saved image to {}.".format(file_path))

    def percep_to_one_hot(self, pred):
        mnist_labels = self.data_provider.percep_mnist_labels().to(self.device)
        result = []
        for prediction in pred:
            for label in mnist_labels:
                result.append(torch.sum(label.eq(prediction)))
        t = torch.stack(result).reshape(pred.shape[0], -1)
        a = t.argmax(1)
        return torch.zeros(t.shape).to(self.device).scatter(1, a.unsqueeze(1), 1.0)

    def normalize(self, x):
        AA = x.clone()
        AA = AA.view(x.size(0), -1)
        AA -= AA.min(1, keepdim=True)[0]
        AA /= AA.max(1, keepdim=True)[0]
        return AA.view(x.size(0), x.size(1), x.size(2), x.size(3))

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