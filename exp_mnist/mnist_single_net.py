import sys
sys.path.insert(1, '/home/gasper/Dokumenti/faks/kognitivna znanost/izmenjava/semester 1/Projekt/ubal-python')
sys.path.insert(1, '/home/j/jelovcan1/UBAL/')
import os
import pickle
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from model.UBAL_torch import UBAL

from exp_mnist.mnist_data_provider import MNISTData
from pathlib import Path

target_type = 'perceptual'
if target_type == 'one_hot':
    from exp_mnist.mnist_UBAL import UBALSim
elif target_type == 'perceptual':
    from exp_mnist.percep_mnist_UBAL import UBALSim

sigmoid = torch.nn.Sigmoid()
softmax = torch.nn.Softmax(dim=1)
act_fun_F = [sigmoid, sigmoid, sigmoid]
act_fun_B = [sigmoid, sigmoid, sigmoid]

hidden_size = 1800
learning_rate = 0.1
init_w_mean = 0.0
init_w_var = 0.5
betas = [1.0, 1.0, 0.9]
# betas = [1.0, 0.9999999, 0.9]
# betas = [1.0, 0.9995, 0.9]
gammasF = [float("nan"), 1.0, 1.0]
gammasB = [1.0, 1.0, float("nan")]
# betas = [0.0, 1.0, 0.0]
# gammasF = [float("nan"), 1.0, 1.0]
# gammasB = [1.0, 1.0, float("nan")]
max_epoch = 120
train_data_size = 55000
validation_data_size = 5000
# train_data_size = 25000
# validation_data_size = 2000

device = "cuda:0"

data_provider = MNISTData(train_size=train_data_size, validation_size=validation_data_size,
                          target_type=target_type,
                          dl_kwargs={'num_workers': 6, 'pin_memory': True})
result_path = str(Path(__file__).parent.parent) + "/results/mnist/"

sim = UBALSim(hidden_size, learning_rate, init_w_mean, init_w_var,
              betas, gammasF, gammasB, max_epoch, data_provider, device, result_path,
              targer_noise_m=0.0, target_noise_variance=0.0005, log_freq=10)
# performance = sim.train_test_one_net(save_net=True, save_backward_images=True, noisy_targets=False)
# performance = sim.train_test_one_net(save_net=True, save_backward_images=True, noisy_targets=True)
performance = sim.train_test_one_net(save_net=True, save_backward_images=True, noisy_targets=False)
# performance = sim.train_test_one_net(save_net=True)



plt.figure()
plt.plot(list(range(max_epoch)),performance["acc_f_train"], label="acc_f_train")
plt.plot(list(range(max_epoch)),performance["acc_f_validation"], label="acc_f_validation")
plt.plot(list(range(max_epoch)),performance["acc_backward_images"], label="acc_backward_images")
# plt.plot(list(range(max_epoch)),performance["mse_b_train"], label="mse_b_train")
plt.savefig("{}single_net_plot.{}.png".format(result_path, int(time.time())))
plt.show()

