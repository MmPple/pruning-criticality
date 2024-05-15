from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
from spikingjelly.clock_driven.functional import reset_net
from spikingjelly.clock_driven import neuron
import pickle

import torch
parser = argparse.ArgumentParser(description='Analysis')
parser.add_argument('--save1', default='', type=str, metavar='PATH',
                    help='path to save pruned model without regeneration (default: none)')
parser.add_argument('--save2', default='', type=str, metavar='PATH',
                    help='path to save pruned model with regeneration (default: none)')
parser.add_argument('--gpu', default=0, type=int,
                    help='gpu id')
parser.add_argument('--arch', default='vgg16', type=str, 
                    help='architecture to use')
args = parser.parse_args()

def find_matching_elements_and_mark_false(tensor1, tensor2):
    if tensor1.dim() != 1 or tensor2.dim() != 1:
        raise ValueError("Wrong Tensor")

    result_tensor1 = torch.ones_like(tensor1, dtype=torch.bool)
    result_tensor2 = torch.ones_like(tensor2, dtype=torch.bool)

    list1 = tensor1.tolist()
    list2 = tensor2.tolist()

    for index, element in enumerate(list1):
        if element in list2:
            result_tensor1[index] = False
            result_tensor2[list2.index(element)] = False

    return result_tensor1, result_tensor2


fp = open(os.path.join(args.save1, f'masks.pkl'), 'rb')
# fp = open('aaai_other_model/vgg16/masks_s0.455_g0.0.pkl', 'rb')
mask1 = pickle.load(fp)
fp.close()

fp = open(os.path.join(args.save2, f'masks.pkl'), 'rb')
# fp = open('aaai_other_model/vgg16/masks_s0.5068_g0.05.pkl', 'rb')
mask2 = pickle.load(fp)
fp.close()

m1only = []
m2only = []
for x1, x2 in zip(mask1, mask2):
    x1only, x2only = find_matching_elements_and_mark_false(torch.where(x1==1)[0], torch.where(x2==1)[0])
    m1only.append(x1only)
    m2only.append(x2only)

fp = os.path.join(args.save1, f'pruned_model.pth.tar')
# fp = 'aaai_other_model/vgg16/pruned_s0.455_g0.0.pth.tar'
checkpoint = torch.load(fp)
if(args.arch == 'resnet19'):
    model1 = checkpoint["model"]
elif(args.arch == 'vgg16'):
    model1 = models.vgg.vgg16_bn_finetune(cfg=checkpoint['cfg'], num_classes=100, total_timestep=5).cuda()
else:
    print("Wrong Arch !")
    exit()
# fp = 'aaai_other_model/vgg16/final_model_0.455/model_best_0.0.pth.tar'
fp = os.path.join(args.save1, f'model_final_best_pruned.pth.tar')
state_dict = torch.load(fp)
model1.load_state_dict(state_dict['state_dict'])

skip = 0
cnt = 0
a1 = 1e6
a2 = 0
for k, m in enumerate(model1.modules()):
    if isinstance(m, nn.BatchNorm2d):
        skip += 1
        w = m.weight.data.clone()
        if(args.arch == "resnet19" and skip % 2 != 0):
            w = w[torch.where(mask1[cnt]==1)[0]]
        if w.size()[0] == 0: 
            cnt += 1
            continue
        if a1 > w.abs().min(): a1 = w.abs().min()
        if a2 < w.abs().max(): a2 = w.abs().max()
        cnt += 1

bn_num = 0
bn_sum = 0
cnt = 0
skip = 0
up_val = a2
down_val = (a1 - a2)

for k, m in enumerate(model1.modules()):
    if isinstance(m, nn.BatchNorm2d):
        skip += 1
        w = m.weight.data.clone()
        if(args.arch == "resnet19" and skip % 2 != 0):
            w = w[torch.where(mask1[cnt]==1)[0]]
        w = (w.abs()-up_val) / down_val 
            # cnt += 1 
            # continue
        # print(w.size(), m1only[cnt].size())
        pbn_num = m1only[cnt].sum().item()
        pbn_sum = w[m1only[cnt]].abs().sum()
        # pbn_num = w.size()[0]
        # pbn_sum = w.abs().sum()

        if(pbn_num > 0):
            bn_num += pbn_num
            bn_sum += pbn_sum
        cnt += 1
print("Mean importance of pruned model 1:", (bn_sum/bn_num).item())

fp = os.path.join(args.save2, f'pruned_model.pth.tar')
# fp = 'aaai_other_model/vgg16/pruned_s0.5068_g0.05.pth.tar'
checkpoint = torch.load(fp)
if(args.arch == 'resnet19'):
    model2 = checkpoint["model"]
elif(args.arch == 'vgg16'):
    model2 = models.vgg.vgg16_bn_finetune(cfg=checkpoint['cfg'], num_classes=100, total_timestep=5).cuda()
else:
    print("Wrong Arch !")
    exit()
fp = os.path.join(args.save2, f'model_final_best_pruned.pth.tar')
# fp = 'aaai_other_model/vgg16/final_model_0.5068/model_best_0.0.pth.tar'
state_dict = torch.load(fp)
model2.load_state_dict(state_dict['state_dict'])

skip = 0
cnt = 0
a1 = 1e6
a2 = 0
for k, m in enumerate(model2.modules()):
    if isinstance(m, nn.BatchNorm2d):
        # if(cnt > 11): continue
        skip += 1
        w = m.weight.data.clone()
        if(args.arch == "resnet19" and skip % 2 != 0):
            w = w[torch.where(mask1[cnt]==1)[0]]
        if w.size()[0] == 0: 
            cnt += 1
            continue
        if a1 > w.abs().min(): a1 = w.abs().min()
        if a2 < w.abs().max(): a2 = w.abs().max()
        cnt += 1

bn_num = 0
bn_sum = 0
cnt = 0
skip = 0
up_val = a2
down_val = (a1 - a2)
for k, m in enumerate(model2.modules()):
    if isinstance(m, nn.BatchNorm2d):
        skip += 1
        w = m.weight.data.clone()
        if(args.arch == "resnet19" and skip % 2 != 0):
            w = w[torch.where(mask2[cnt]==1)[0]]
        w = (w.abs()-up_val) / down_val
            # cnt += 1 
            # continue
        # print(w.size(), m2only[cnt].size())
        pbn_num = m2only[cnt].sum().item()

        pbn_sum = w[m2only[cnt]].abs().sum()
        # pbn_num = w.size()[0]
        # pbn_sum = w.abs().sum()
        if(pbn_num > 0):
            bn_num += pbn_num
            bn_sum += pbn_sum
            # print(cnt, pbn_sum.item(), pbn_num, (pbn_sum/pbn_num).item())
        cnt += 1
print("Mean importance of pruned model 2:", (bn_sum/bn_num).item())