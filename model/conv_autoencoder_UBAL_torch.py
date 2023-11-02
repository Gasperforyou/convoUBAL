import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import sys
torch.set_printoptions(threshold = 100000)
torch.manual_seed(1417)
class ConvAutoencoderUBAL(nn.Module):
    def __init__(self):
        super(ConvAutoencoderUBAL, self).__init__()
        c = 1
        self.conv1F = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)  # out: c x 14 x 14
        self.conv1F.bias = nn.Parameter(torch.ones(1))
        self.conv1F.weight = nn.Parameter((torch.ones(4, 4)*0.35).view(1, 1, 4, 4))
        self.conv1F.weight.detach()
        self.conv1B = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.conv1B.bias = nn.Parameter(torch.ones(1))
        self.conv1B.weight = nn.Parameter((torch.ones(4, 4)*0.35).view(1, 1, 4, 4))
        self.conv1B.weight.detach()
        self.conv2F = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7
        self.conv2F.bias = nn.Parameter(torch.ones(1))
        self.conv2F.weight = nn.Parameter((torch.ones(4, 4)*0.35).view(1, 1, 4, 4))
        self.conv2F.weight.detach()
        self.conv2B = nn.ConvTranspose2d(in_channels=c, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2B.bias = nn.Parameter(torch.ones(1))
        self.conv2B.weight = nn.Parameter((torch.ones(4, 4)*0.35).view(1, 1, 4, 4))
        self.conv2B.weight.detach()
        self.conv3F = nn.ConvTranspose2d(in_channels=c, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv3F.bias = nn.Parameter(torch.ones(1))
        self.conv3F.weight = nn.Parameter((torch.ones(4, 4)*0.35).view(1, 1, 4, 4))
        self.conv3F.weight.detach()
        self.conv3B = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7
        self.conv3B.bias = nn.Parameter(torch.ones(1))
        self.conv3B.weight = nn.Parameter((torch.ones(4, 4)*0.35).view(1, 1, 4, 4))
        self.conv3B.weight.detach()
        self.conv4F = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.conv4F.bias = nn.Parameter(torch.ones(1))
        self.conv4F.weight = nn.Parameter((torch.ones(4, 4)*0.35).view(1, 1, 4, 4))
        self.conv4F.weight.detach()
        self.conv4B = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)  # out: c x 14 x 14
        self.conv4B.bias = nn.Parameter(torch.ones(1))
        self.conv4B.weight = nn.Parameter((torch.ones(4, 4)*0.35).view(1, 1, 4, 4))
        self.conv4B.weight.detach()

    def forward(self, input_x, input_y):
        act_FP = [None] * 5
        act_BP = [None] * 5
        act_FE = [None] * 5
        act_BE = [None] * 5
        relu = torch.nn.ReLU()
        act_FP[0] = input_x
        act_FP[1] = relu(self.conv1F(act_FP[0]))
        act_FE[0] = relu(self.conv1B(act_FP[1]))
        act_FP[2] = relu(self.conv2F(act_FP[1]))
        act_FE[1] = relu(self.conv2B(act_FP[2]))
        act_FP[3] = relu(self.conv3F(act_FP[2]))
        act_FE[2] = relu(self.conv3B(act_FP[3]))
        act_FP[4] = torch.sigmoid(self.conv4F(act_FP[3]))
        act_FE[3] = relu(self.conv4B(act_FP[4]))

        act_BP[4] = input_y
        act_BP[3] = relu(self.conv4B(act_BP[4]))
        act_BE[4] = relu(self.conv4F(act_BP[3]))
        act_BP[2] = relu(self.conv3B(act_BP[3]))
        act_BE[3] = relu(self.conv3F(act_BP[2]))
        act_BP[1] = relu(self.conv2B(act_BP[2]))
        act_BE[2] = relu(self.conv2F(act_BP[1]))
        act_BP[0] = torch.sigmoid(self.conv1B(act_BP[1]))
        act_BE[1] = relu(self.conv1F(act_BP[0]))

        if(torch.all(torch.isnan(act_FP[4]))):
            print("Nan found")
            sys.exit()
        return act_FP, act_BP, act_FE, act_BE

class UBALoptim(Optimizer):
    def __init__(self, params, betas, gammas_f, gammas_b, device, lr=required):
        defaults = dict(lr=lr, betas=betas, gammas_f=gammas_f, gammas_b=gammas_b,
                        device=device)
        super(UBALoptim, self).__init__(params, defaults)

    def step(self, act_FP, act_BP, act_FE, act_BE):
        for group in self.param_groups:

            # for every layer unfold the activations

            size_FP = [None] * len(act_FP)
            size_BP = [None] * len(act_BP)
            act_FP_unfolded = [None] * len(act_FP)
            act_BP_unfolded = [None] * len(act_BP)
            act_FE_unfolded = [None] * len(act_FE)
            act_BE_unfolded = [None] * len(act_BE)
            for i in range(len(act_FP)):
                size_FP[i] = (act_FP[i].size()[2], act_FP[i].size()[3])
                size_BP[i] = (act_BP[i].size()[2], act_BP[i].size()[3])
                unfold = nn.Unfold(kernel_size=4, dilation = 1, stride=2, padding=1)
                act_FP_unfolded[i] = unfold(act_FP[i])
                act_BP_unfolded[i] = unfold(act_BP[i])
                if(i<len(act_FE)-1):
                    act_FE_unfolded[i] = unfold(act_FE[i])
                if (i > 0):
                    act_BE_unfolded[i] = unfold(act_BE[i])


            # calculate targets and estimates, perfrom wight change equation and apply them to kernel weights
            for p in range(int(len(group['params'])/4)):

                if(size_FP[p][0]*size_FP[p][1]>=size_FP[p+1][0]*size_FP[p+1][1]):
                    target_F = (act_FP[p+1].mul(group['betas'][p+1])).add(act_BP[p+1].mul(1.0 - group['betas'][p+1])).to(group['device'])
                    target_B = (act_BP_unfolded[p].mul(1.0-group['betas'][p])).add(act_FP_unfolded[p].mul(group['betas'][p])).to(group['device'])
                    estimateF = (act_FP[p+1].mul(group['gammas_f'][p+1])).add(act_BE[p+1].mul(1.0 - group['gammas_f'][p+1])).to(group['device'])
                    estimateB = (act_BP_unfolded[p].mul(group['gammas_b'][p])).add(act_FE_unfolded[p].mul(1.0 - group['gammas_b'][p])).to(group['device'])

                    # Weight change equation

                    # weight_update_F
                    vector = target_F.sub(estimateF).flatten(start_dim=1)
                    outputs = []
                    for i in range(target_B.size(0)):
                        outputs.append((target_B[i]*vector[i]).mul(group['lr']).to(group['device']))
                    weight_update_F = torch.stack(outputs).sum((0, 2))/len(outputs)/outputs[0].size(1)

                    # weight_update_B
                    vector = target_B.sub(estimateB)
                    target_F = target_F.flatten(start_dim=1)
                    outputs = []
                    for i in range(target_F.size(0)):
                        outputs.append((target_F[i] * vector[i]).mul(group['lr']).to(group['device']))
                    weight_update_B = torch.stack(outputs).sum((0, 2))/len(outputs)/outputs[0].size(1)

                else:
                    target_F = (act_FP_unfolded[p + 1].mul(group['betas'][p + 1])).add(
                        act_BP_unfolded[p + 1].mul(1.0 - group['betas'][p + 1])).to(group['device'])
                    target_B = (act_BP[p].mul(1.0 - group['betas'][p])).add(
                        act_FP[p].mul(group['betas'][p])).to(group['device'])
                    estimateF = (act_FP_unfolded[p + 1].mul(group['gammas_f'][p + 1])).add(
                        act_BE_unfolded[p + 1].mul(1.0 - group['gammas_f'][p + 1])).to(group['device'])
                    estimateB = (act_BP[p].mul(group['gammas_b'][p])).add(
                        act_FE[p].mul(1.0 - group['gammas_b'][p])).to(group['device'])

                    # Weight change equation

                    # weight_update_F
                    vector = target_F.sub(estimateF)
                    target_B_0 = target_B.flatten(start_dim=1)
                    outputs = []
                    for i in range(target_B.size(0)):
                        outputs.append((target_B_0[i] * vector[i]).mul(group['lr']).to(group['device']))
                    weight_update_F = torch.stack(outputs).sum((0, 2))/len(outputs)/outputs[0].size(1)

                    # weight_update_B
                    vector = target_B.sub(estimateB).flatten(start_dim=1)
                    outputs = []
                    for i in range(target_F.size(0)):
                        outputs.append((target_F[i] * vector[i]).mul(group['lr']).to(group['device']))
                    weight_update_B = torch.stack(outputs).sum((0, 2))/len(outputs)/outputs[0].size(1)

                #print(group['params'][p*4].data)
                # apply the changes
                group['params'][p*4].data.add_(weight_update_F.view(4, 4))
                group['params'][p*4+2].data.add_(weight_update_B.view(4, 4))





