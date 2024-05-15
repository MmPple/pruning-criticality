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
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg16', type=str, 
                    help='architecture to use')
parser.add_argument('--timestep', default=5, type=int,
                    help='timestep for SNN')
parser.add_argument('--gpu', default=0, type=int,
                    help='gpu id')
parser.add_argument('--show', action='store_true', default=False,
                    help='show loss and accuracy list')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../dataset/cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../dataset/cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    n_class = 10
elif args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../dataset/cifar100', train=True, download=True,
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

else:
    if(args.arch == "vgg16"):
        model = models.vgg.vgg16_bn(num_classes=n_class, total_timestep=args.timestep).cuda()
    elif(args.arch == "resnet19"): 
        model = models.resnet.ResNet19(num_classes=n_class, total_timestep=args.timestep, 
            pruning=False, v_threshold=1.0)
    else:
        print("Wrong Arch!")
        exit()

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if(m.weight.grad is not None):
                m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def train(epoch):
    model.train()
    losses = []
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output_list, _ = model(data)
        loss = 0
        for output in output_list:
            loss += F.cross_entropy(output, target) / args.timestep
        pred = sum(output_list).data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        reset_net(model)
        losses.append(loss.item())

        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return correct.item() / float(len(train_loader.dataset)), sum(losses) / len(losses)

def test(data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_spikes = 0
    with torch.no_grad():
      for data, target in data_loader:
          if args.cuda:
              data, target = data.cuda(), target.cuda()
          data, target = Variable(data, volatile=True), Variable(target)
          output, v_list = model(data)
        for (s, v) in v_list:
            total_spikes += s.detach().sum().item()
          output = sum(output)
          loss = F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
          test_loss += loss
          pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
          correct += pred.eq(target.data.view_as(pred)).cpu().sum()
          reset_net(model)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    total_spikes /= len(test_loader.dataset)
    print('\n Total Spikes: ', total_spikes)
    return correct.item() / float(len(test_loader.dataset)), test_loss

def save_checkpoint(state, is_best, args, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        if(args.refine):
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, f'model_final_best_pruned.pth.tar'))
        else:
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

best_prec1 = 0.
train_records = {"loss":[], "acc":[]}
test_records = {"loss":[], "acc":[]}

for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    
    train_acc, train_loss = train(epoch)
    train_records["loss"].append(train_loss)
    train_records["acc"].append(100*train_acc)

    prec1, test_loss = test(test_loader)
    test_records["loss"].append(test_loss)
    test_records["acc"].append(100*prec1)
    
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, args, filepath=args.save)

print("Best accuracy: "+str(best_prec1))

if(args.show):
    print()
    print("Train Loss List:",  train_records["loss"])
    print("Train Accuracy List:",  train_records["acc"])
    print("Test Loss List:",  test_records["loss"])
    print("Test Accuracy List:",  test_records["acc"])
