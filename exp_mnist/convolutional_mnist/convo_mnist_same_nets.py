import sys
sys.path.insert(1, '/home/gasper/Dokumenti/faks/kognitivna znanost/izmenjava/semester 1/Projekt/ubal-python')

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from exp_mnist.convolutional_mnist.convo_mnist_UBAL import UBALSim
from exp_mnist.mnist_data_provider import MNISTData

param_learning_rate = 0.03
param_init_w_mean = 0.0
param_init_w_var = 0.5
param_betas = [1.0, 1.0, 1.0, 0.5]
param_gammasF = [float("nan"), 1.0, 1.0, 1.0]
param_gammasB = [0.0, 0.0, 0.0, float("nan")]
# param_betas = [0.0, 1.0, 0.0]
# param_gammasF = [float("nan"), 1.0, 1.0]
# param_gammasB = [1.0, 1.0, float("nan")]
# train_data_size = 5000
# validation_data_size = 1000
# minibatch_size = 256
param_max_epoch = 120
netcount = 5

settings_conv = [{"in_channels":1, "out_channels":1, "kernel_size":5, "stride":2, "padding":0},
            {"in_channels":1, "out_channels":1, "kernel_size":4, "stride":1, "padding":0}]
settings_trans = [{"in_channels":1, "out_channels":1, "kernel_size":5, "stride":2, "padding":0},
            {"in_channels":1, "out_channels":1, "kernel_size":4, "stride":1, "padding":0}]
sigmoid = torch.nn.Sigmoid()
softmax = torch.nn.Softmax(dim=1)
relu = torch.nn.ReLU()
act_fun_F = [relu, relu, relu]
act_fun_B = [relu, relu, relu]

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
kwargs = {'num_workers': 6, 'pin_memory': True} if 'cuda' in device else {}
print("Using {} with {}".format(device, kwargs))

data_provider = MNISTData()
result_path = str(Path(__file__).parent.parent.parent) + "/results/mnist/"

sim_start_time = time.time()
perf_train = np.zeros((netcount, param_max_epoch))
perf_valid = np.zeros((netcount, param_max_epoch))
perf_b_train = np.zeros((netcount, param_max_epoch))
# perf_test = np.zeros((netcount, param_max_epoch))
final_train = np.zeros(netcount)
final_valid = np.zeros(netcount)
final_test = np.zeros(netcount)
final_b_train = np.zeros(netcount)

for n in range(netcount):
    print("Training net ", n+1)
    sim = UBALSim(settings_conv, settings_trans, param_learning_rate, param_init_w_mean, param_init_w_var,
                  param_betas, param_gammasF, param_gammasB, param_max_epoch, act_fun_F, act_fun_B, data_provider, device, result_path)
    performance = sim.train_test_one_net(save_net=True, save_backward_images=True, noisy_targets=False)
    perf_train[n] = performance["acc_f_train"]
    perf_valid[n] = performance["acc_f_validation"]
    perf_b_train[n] = performance["mse_b_train"]
    # perf_test[n] = performance["test"]
    final_train[n] = performance["acc_f_train"][param_max_epoch-1]
    final_valid[n] = performance["acc_f_validation"][param_max_epoch-1]
    final_test[n] = performance["acc_f_test"][0]
    final_b_train[n] = performance["mse_b_train"][param_max_epoch - 1]
sim_end_time = time.time()

results_file_name = 'results_mnist_{}.npz'.format(sim_end_time)
with open(results_file_name, 'wb') as f:
    np.savez(f, perf_train)
    np.savez(f, perf_valid)
    # np.savez(f, perf_test)

    # print("Accuracy Test: {:.3f}%".format(performance["test"] * 100))
    # print()
    print("Total training time: {} seconds".format(sim_end_time - sim_start_time))
    # print()
    print("Average final performance train: {:.4f} validation: {:.4f} test: {:.4f} backward: {:.4f}".format(
        np.mean(final_train),np.mean(final_valid),np.mean(final_test), np.mean(final_b_train)))
    print("Maximum final performance train: {:.4f} validation: {:.4f} test: {:.4f} backward: {:.4f}".format(
        np.max(final_train),np.max(final_valid),np.max(final_test), np.max(final_b_train)))
    # print()
    print("Overal final performance")
    print("train: ",final_train)
    print("validation: ",final_valid)
    print("test: ",final_test)
    print("backward: ",final_b_train)
    print()

# plt.plot(list(range(sim.train_epochs)), np.mean(performance["train"], axis=0))
# plt.plot(list(range(sim.train_epochs)), np.mean(performance["validation"], axis=0))
# plt.plot(list(range(sim.train_epochs)), np.mean(performance["test"], axis=0))
# plt.show()