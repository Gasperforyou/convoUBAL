import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import sys
torch.set_printoptions(threshold = 100000)
class ConvUBAL(nn.Module):
    def __init__(self, settings_conv, settings_trans, act_fun_f, act_fun_b, device):
        super(ConvUBAL, self).__init__()
        self.layers = len(settings_conv)+1
        self.act_fun_f = act_fun_f
        self.act_fun_b = act_fun_b
        self.device = device

        # create neural network full of convolutions
        # create in forward direction of convolution layers
        self.net = nn.ModuleList()
        for i in range(len(settings_conv)):
            self.net.append(nn.Conv2d(**settings_conv[i]))
            self.net[2*i].bias = nn.Parameter(torch.zeros(1))
            #self.net_f[i].weight = nn.Parameter((torch.ones(4, 4) * 0.35).view(1, 1, 4, 4))
            self.net[2*i].weight.requires_grad = False
            self.net[2*i].bias.requires_grad = False

            self.net.append(nn.ConvTranspose2d(**settings_trans[i]))
            self.net[2*i+1].bias = nn.Parameter(torch.zeros(1))
            # self.net_b[i].weight = nn.Parameter((torch.ones(4, 4) * 0.35).view(1, 1, 4, 4))
            self.net[2*i+1].weight.requires_grad = False
            self.net[2*i+1].bias.requires_grad = False
        self.net.to(self.device)
    def forward(self, input_x):

        # compute activations of predictions and echos in forward direction
        act_FP = [None] * self.layers
        act_FE = [None] * self.layers

        act_FP[0] = input_x
        for i in range(1, self.layers):
            act_FP[i] = self.act_fun_f[i](self.net[2*(i-1)](act_FP[i-1]))
            act_FE[i-1] = self.act_fun_b[i-1](self.net[2*(i-1)+1](act_FP[i]))

        # check if there is nan then terminate
        if(torch.all(torch.isnan(act_FP[self.layers-1]))):
            print("Nan found")
            sys.exit()
        return act_FP, act_FE

    def backward(self, target):
        act_BP = [None] * self.layers
        act_BE = [None] * self.layers

        act_BP[self.layers-1] = target
        for i in range(self.layers-1, 0, -1):
            act_BP[i-1] = self.act_fun_f[i-1](self.net[2*(i-1)+1](act_BP[i]))
            act_BE[i] = self.act_fun_b[i](self.net[2*(i-1)](act_BP[i-1]))

        if (torch.all(torch.isnan(act_BP[0]))):
            print("Nan found")
            sys.exit()
        return act_BP, act_BE
class UBALoptim(Optimizer):
    def __init__(self, params, betas, gammas_f, gammas_b, device, settings_conv, settings_trans, lr=required):
        defaults = dict(lr=lr, betas=betas, gammas_f=gammas_f, gammas_b=gammas_b,
                        device=device, settings_conv = settings_conv, settings_trans=settings_trans)
        super(UBALoptim, self).__init__(params, defaults)

    def step(self, act_FP, act_FE, act_BP, act_BE):
        for group in self.param_groups:
            # for every layer unfold the activations
            act_FP_unfolded = [None] * len(act_FP)
            act_BP_unfolded = [None] * len(act_BP)
            act_FE_unfolded = [None] * len(act_FE)
            act_BE_unfolded = [None] * len(act_BE)

            # unfold appropriate activations
            for i in range(len(act_FP)-1):
                settings_conv = {k: group['settings_conv'][i][k] for k in group['settings_conv'][i].keys() - {'out_channels', 'in_channels'}}
                settings_trans = {k: group['settings_trans'][i][k] for k in group['settings_trans'][i].keys() - {'out_channels', 'in_channels'}}
                unfold_conv = nn.Unfold(**settings_conv)
                unfold_trans = nn.Unfold(**settings_trans)
                act_FP_unfolded[i] = unfold_conv(act_FP[i])
                act_BP_unfolded[i] = unfold_trans(act_BP[i])
                act_FE_unfolded[i] = unfold_trans(act_FE[i])

            # calculate targets and estimates, perfrom wight change equation and apply them to kernel weights
            for p in range(int(len(group['params'])/4)):
                target_F = (act_FP[p+1].mul(group['betas'][p+1])).add(act_BP[p+1].mul(1.0 - group['betas'][p+1]))
                target_B = (act_BP_unfolded[p].mul(1.0-group['betas'][p])).add(act_FP_unfolded[p].mul(group['betas'][p]))
                estimateF = (act_FP[p+1].mul(group['gammas_f'][p+1])).add(act_BE[p+1].mul(1.0 - group['gammas_f'][p+1]))
                estimateB = (act_BP_unfolded[p].mul(group['gammas_b'][p])).add(act_FE_unfolded[p].mul(1.0 - group['gammas_b'][p]))

                # Weight change equation

                # weight_update_F
                vector = target_F.sub(estimateF).flatten(start_dim=1)
                outputs = []
                for i in range(target_B.size(0)):
                    outputs.append((target_B[i]*vector[i]).mul(group['lr']))
                weight_update_F = torch.stack(outputs).sum((0, 2))/len(outputs)/outputs[0].size(1)

                # weight_update_B
                vector = target_B.sub(estimateB)
                target_F = target_F.flatten(start_dim=1)
                outputs = []
                for i in range(target_F.size(0)):
                    outputs.append((target_F[i] * vector[i]).mul(group['lr']))

                weight_update_B = torch.stack(outputs).sum((0, 2))/len(outputs)/outputs[0].size(1)

                # apply the changes
                group['params'][p*4].data += (weight_update_F.view(group['settings_conv'][p]["kernel_size"], group['settings_conv'][p]["kernel_size"]))
                group['params'][p*4+2].data += (weight_update_B.view(group['settings_trans'][p]["kernel_size"], group['settings_trans'][p]["kernel_size"]))