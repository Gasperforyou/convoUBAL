import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import sys


from model.conv_UBAL_torch import ConvUBAL, UBALoptim
from model.UBAL_torch import UBAL

class ConvNet:
    def __init__(self, settings_conv, settings_trans, act_fun_F, act_fun_B,  learning_rate, betas, gammasF, gammasB, device):
        self.act_fun_F = act_fun_F
        self.act_fun_B = act_fun_B
        self.settings_conv = settings_conv
        self.settings_trans = settings_trans
        self.learning_rate = learning_rate
        self.betas = betas
        self.gammasF = gammasF
        self.gammasB = gammasB
        self.device = device
        self.model = nn.ModuleList()
        self.model.append(ConvUBAL(self.settings_conv, self.settings_trans, self.act_fun_F, self.act_fun_B, self.device))
        self.optimizer = UBALoptim(self.model[0].parameters(), self.betas, self.gammasF, self.gammasB, self.device,
                              self.settings_conv, self.settings_trans, lr=self.learning_rate)
        self.model.append(UBAL([81, 10], [torch.nn.ReLU(), torch.nn.Softmax(dim=1)],
                          [torch.nn.ReLU(), torch.nn.Softmax(dim=1)], self.learning_rate, 0.0, 0.5,
                          self.betas[-2:],
                          self.gammasF[-2:], self.gammasB[-2:], self.device))

    def train(self):
        self.model[0].train()
    def eval(self):
        self.model[0].eval()

    def activation(self, images, labels):
        act_fp_conv, act_fe_conv = self.model[0].forward(images)
        act_fp_conv_flat = act_fp_conv[self.model[0].layers - 1].view(-1, 81)
        act_fp_normal, act_fe_normal, act_bp_normal, act_be_normal = self.model[1].activation(act_fp_conv_flat, labels)
        act_bp_normal_rectangle = act_bp_normal[0].view(-1, 1, 9, 9)
        act_bp_conv, act_be_conv = self.model[0].backward(act_bp_normal_rectangle)
        return act_fp_conv, act_fe_conv, act_bp_conv, act_be_conv, act_fp_normal, act_fe_normal, act_bp_normal, act_be_normal

    def learn(self, act_fp, act_fe, act_bp, act_be, act_fp_normal, act_fe_normal, act_bp_normal, act_be_normal):
        self.optimizer.step(act_fp, act_fe, act_bp, act_be)
        self.model[1].learning(act_fp_normal, act_fe_normal, act_bp_normal, act_be_normal)


    def activation_FP_last(self, input_x):
        act_fp_conv, act_fe_conv = self.model[0].forward(input_x)
        act_fp_conv_flat = act_fp_conv[self.model[0].layers - 1].view(-1, 81)
        act_fp = self.model[1].activation_FP_last(act_fp_conv_flat)
        return act_fp
    def activation_BP_last(self,  input_y):
        act_bp = self.model[1].activation_BP_last(input_y)
        act_bp_normal_rectangle = act_bp.view(-1, 1, 9, 9)
        act_bp_conv, _ = self.model[0].backward(act_bp_normal_rectangle)
        return act_bp_conv[0]

    def forward(self, input_x):
        return self.model[0].forward(input_x)