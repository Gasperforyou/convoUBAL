import sys
sys.path.insert(1, '/home/gasper/Dokumenti/faks/kognitivna znanost/izmenjava/semester 1/Projekt/ubal-python')
import os
import pickle
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


from exp_mnist.mnist_data_provider import MNISTData
from pathlib import Path

from exp_mnist.convolutional_mnist.convo_mnist_UBAL import UBALSim

target_type = 'one-hot'
sigmoid = torch.nn.Sigmoid()
softmax = torch.nn.Softmax(dim=1)
relu = torch.nn.ReLU()
act_fun_F = [relu, relu, relu]
act_fun_B = [relu, relu, relu]

settings_conv = [{"in_channels":1, "out_channels":1, "kernel_size":5, "stride":2, "padding":0},
            {"in_channels":1, "out_channels":1, "kernel_size":4, "stride":1, "padding":0}]
settings_trans = [{"in_channels":1, "out_channels":1, "kernel_size":5, "stride":2, "padding":0},
            {"in_channels":1, "out_channels":1, "kernel_size":4, "stride":1, "padding":0}]
learning_rate = 0.03
init_w_mean = 0.0
init_w_var = 0.5
betas = [1.0, 1.0, 1.0, 0.5]

gammasF = [float("nan"), 1.0, 1.0, 1.0]
gammasB = [0.0, 0.0, 0.0, float("nan")]
max_epoch = 120
train_data_size = 55000
validation_data_size = 5000
minibatch_size = 265
# train_data_size = 25000
# validation_data_size = 2000

torch.manual_seed(10)

device = "cuda:0"

data_provider = MNISTData(train_size=train_data_size, validation_size=validation_data_size,
                          target_type=target_type,
                          minibatch_size = minibatch_size,
                          dl_kwargs={'num_workers': 6, 'pin_memory': True})
result_path = str(Path(__file__).parent.parent.parent) + "/results/mnist/"

sim = UBALSim(settings_conv, settings_trans, learning_rate, init_w_mean, init_w_var,
              betas, gammasF, gammasB, max_epoch, act_fun_F, act_fun_B, data_provider, device, result_path,
              target_noise_m=0.0, target_noise_variance=0.0005, log_freq=10)
# performance = sim.train_test_one_net(save_net=True, save_backward_images=True, noisy_targets=False)
# performance = sim.train_test_one_net(save_net=True, save_backward_images=True, noisy_targets=True)
performance = sim.train_test_one_net(save_net=True, save_backward_images=False, noisy_targets=False)
# performance = sim.train_test_one_net(save_net=True)



plt.figure()
plt.plot(list(range(max_epoch)),performance["acc_f_train"], label="acc_f_train")
plt.plot(list(range(max_epoch)),performance["acc_f_validation"], label="acc_f_validation")
plt.plot(list(range(max_epoch)),performance["acc_backward_images"], label="acc_backward_images")
# plt.plot(list(range(max_epoch)),performance["mse_b_train"], label="mse_b_train")
plt.savefig("{}single_net_plot.{}.png".format(result_path, int(time.time())))
plt.show()

