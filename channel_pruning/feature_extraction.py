'''
Modified based on the code of Learning Efficient Convolutional Networks Through Network Slimming
'''
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
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--final_model', default='./logs', type=str, metavar='PATH',
                    help='path to save final model ')
parser.add_argument('--arch', default='vgg16', type=str, 
                    help='architecture to use')
parser.add_argument('--timestep', default=5, type=int,
                    help='timestep for SNN')
parser.add_argument('--gpu', default=0, type=int,
                    help='gpu id')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data/cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data/cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    n_class = 10
elif args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../dataset/cifar100', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../dataset/cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    n_class = 100

if args.refine:
    checkpoint = torch.load(args.refine)
    print(checkpoint['cfg'])
    if(args.arch == "vgg16"):
        model = models.vgg.vgg16_bn_finetune(cfg=checkpoint['cfg'], num_classes=n_class, total_timestep=args.timestep).cuda()
        model.load_state_dict(checkpoint['state_dict'])
    elif(args.arch == "resnet19"):
        model = checkpoint['model']
    else:
        print("Wrong Arch!")
        exit()
    state_dict = torch.load(args.final_model)["state_dict"]
    model.load_state_dict(state_dict)
else:
    print("Need model!")
    exit()

if args.cuda:
    model.cuda()


def test(data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    s_map = [[] for i in range(100)]
    with torch.no_grad():
      for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, v_list = model(data)
        output = sum(output)
        loss = F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        test_loss += loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        reset_net(model)

        (spike_list,v_list) = v_list
        s_list = []
        for t in range(len(spike_list)):
            for l in range(len(spike_list[t])):
                if(t==0):
                    s_list.append(spike_list[t][l])
                else:
                    s_list[l] += spike_list[t][l]
        for l in range(len(s_list)):
            if l < (len(s_list) -1): continue
            # print(len(s_map[0]))
            if(len(s_list[l].size()) > 2):
                # if(l == 2):
                # print(s_list[l].sum(axis=[2,3])[indexs])
                s_t = s_list[l].sum(axis=[2,3])
                if(s_t.size()[-1] == 0): break
                s_t = (s_t.max(dim=1, keepdim=True)[0] - s_t) / (s_t.max(dim=1, keepdim=True)[0]-s_t.min(dim=1, keepdim=True)[0])
                for i, k in enumerate(target):
                    index = k.item()
                    # if l >= len(s_map[index]):
                    if len(s_map[index]) == 0:
                        s_map[index].append(s_t[[i]])
                    else:
                        # s_map[index][] = torch.cat([s_map[index][l], s_t[[i]]], dim=0)
                        s_map[index][0] = torch.cat([s_map[index][0], s_t[[i]]], dim=0)
        print(s_map[3][0].size())
    # means = [0 for i in range(len(s_map[0]))]
    means = []
    for index, ss in enumerate(s_map):
        means.append([0, 0])
        for i, s in enumerate(ss):
            v = s.var(dim=0)
            s = s.mean(dim=0)
            means[index][1] = v.mean().item()
            means[index][0] = s
    return correct.item() / float(len(test_loader.dataset)), means


prec, train_means = test(train_loader)
prec1, test_means = test(test_loader)
print("Intra-cluster variance of classes for Train set: ")
print([s[1] for s in train_means])
print("Intra-cluster variance of classes for Test set: ")
print([s[1] for s in test_means])
cs = []
for i, (s1, s2) in enumerate(zip(train_means, test_means)):  
    cs.append(torch.cosine_similarity(s1[0], s2[0], dim=0).item())
print("Cosine Similarity for classes between trainset and testset:")
print(cs)
