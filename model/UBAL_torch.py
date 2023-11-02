import pickle

import torch
import torch.nn


class UBAL(torch.nn.Module):

    # def __init__(self, layers, activation_funcions, learning_rate=0.02, init_w_mean=0.0, init_w_variance=0.02):
    def __init__(self, layers, act_fun_f, act_fun_b, learning_rate, init_w_mean, init_w_variance,
                 betas, gammas_f, gammas_b, device):
        super(UBAL, self).__init__()
        self.device = device
        self.arch = layers
        # depth
        self.d = len(self.arch)
        self.act_fun_F = act_fun_f
        self.act_fun_B = act_fun_b
        self.learning_rate = learning_rate
        self.init_weight_mean = init_w_mean
        self.init_weight_variance = init_w_variance
        self.betas = torch.FloatTensor(betas).to(self.device)
        self.gammasF = torch.FloatTensor(gammas_f).to(self.device)
        self.gammasB = torch.FloatTensor(gammas_b).to(self.device)
        self.weightsF = []
        for i in range(self.d - 1):
            self.weightsF.append(self.create_init_weights_with_bias(self.arch[i], self.arch[i + 1]))
        self.weightsB = []
        for i in range(self.d - 1):
            self.weightsB.append(self.create_init_weights_with_bias(self.arch[i + 1], self.arch[i]))

    def create_init_weights_with_bias(self, dimx, dimy):
        weights = torch.normal(mean=self.init_weight_mean, std=self.init_weight_variance, size=(dimx + 1, dimy)).to(self.device)
        # weights = torch.empty(dimx + 1, dimy)
        # torch.nn.init.xavier_normal_(weights)
        return weights

    def add_bias(self, input_array):
        bias_padding = torch.ones(input_array.shape[0], 1).to(self.device)
        output_array = torch.hstack((input_array, bias_padding))
        return output_array

    # assume x and y are matrices with minibatches, first dim is minibatch size
    def activation_FP_last(self, input_x):
        act_FP = input_x
        for i in range(1, self.d):
            act_FP = self.act_fun_F[i](self.add_bias(act_FP).matmul(self.weightsF[i - 1]))
        return act_FP

    def activation_BP_last(self, input_y):
        act_BP = input_y
        for i in range(self.d - 1, 0, -1):
            act_BP = self.act_fun_F[i](self.add_bias(act_BP).matmul(self.weightsB[i - 1]))
        return act_BP

    # assume x and y are matrices with minibatches, first dim is minibatch size
    # assume x and y were already put to the device!!
    def activation(self, input_x, input_y):
        act_FP = [None] * self.d
        act_BP = [None] * self.d
        act_FE = [None] * self.d
        act_BE = [None] * self.d

        act_FP[0] = input_x
        for i in range(1, self.d):
            act_FP[i] = self.act_fun_F[i](self.add_bias(act_FP[i - 1]).matmul(self.weightsF[i - 1]))
            act_FE[i - 1] = self.act_fun_B[i](self.add_bias(act_FP[i]).matmul(self.weightsB[i - 1]))

        act_BP[self.d - 1] = input_y
        for i in range(self.d - 1, 0, -1):
            act_BP[i - 1] = self.act_fun_F[i](self.add_bias(act_BP[i]).matmul(self.weightsB[i - 1]))
            act_BE[i] = self.act_fun_B[i](self.add_bias(act_BP[i - 1]).matmul(self.weightsF[i - 1]))

        # return {'FP': act_FP, 'FE': act_FE, 'BP': act_BP, 'BE': act_BE}
        return act_FP, act_FE, act_BP, act_BE

    # assume activations in the whole net as dict indexed by strings to avoid ordering confusion
    def learning(self, act_FP, act_FE, act_BP, act_BE):
        target = [None] * self.d
        target_with_bias = [None] * self.d
        estimateF = [None] * self.d
        estimateB = [None] * self.d

        for l in range(len(self.arch)):
            # t_q = beta^F_q q^FP + (1 - beta^F_q) q^BP
            target[l] = act_FP[l].mul(self.betas[l]).add(act_BP[l].mul(1.0 - self.betas[l]))
            target_with_bias[l] = self.add_bias(target[l])
            if l > 0:
                # e^F_q = gamma^F_q q^FP + (1 − gamma^F_q) q^BE
                estimateF[l] = act_FP[l].mul(self.gammasF[l]).add(act_BE[l].mul(1.0 - self.gammasF[l]))
            if l < (self.d - 1):
                # e^B_p = gamma^B p^BP + (1 - gamma^B_p) p^FE
                estimateB[l] = act_BP[l].mul(self.gammasB[l]).add(act_FE[l].mul(1.0 - self.gammasB[l]))

        for l in range(self.d - 1):
            k = l + 1
            # todo optimize
            weight_update_F = target_with_bias[l].transpose(0, 1).matmul(
                target[k].sub(estimateF[k])
            ).mul(self.learning_rate)
            weight_update_B = target_with_bias[k].transpose(0, 1).matmul(
                target[l].sub(estimateB[l])
            ).mul(self.learning_rate)
            # print("WC_F[{}]: {} {}".format(l, weight_update_F.size(), weight_update_F[1]))
            # print("WC_B[{}]: {} {}".format(l, WCB.size(), WCB[1]))
            # print("weight_F[{}]: {}".format(l, self.weightsF[l].numpy()))
            # print("DELTAw_F[{}]: {}".format(l, weight_update_F.numpy()))
            # print("weight_B[{}]: {}".format(l, self.weightsB[l].numpy()))
            # print("DELTAw_B[{}]: {}".format(l, weight_update_B.numpy()))
            self.weightsF[l].add_(weight_update_F)
            self.weightsB[l].add_(weight_update_B)

    def load_weights(self, file_path):
        with open(file_path, 'rb') as handle:
            wts_all = pickle.load(handle)
            for i in range(self.d-1):
                self.weightsF[i] = wts_all[i]
                self.weightsB[i] = wts_all[i+self.d-1]

    def save_weights(self, file_path):
        wts_all = self.weightsF + self.weightsB
        with open(file_path, 'wb') as handle:
            pickle.dump(wts_all, handle)
            print("Saved weights to {}.".format(file_path))