'''
Modified based on the code of Exploring Lottery Ticket Hypothesis in Spiking Neural Networks
'''
import time
import utils
import config
import copy
import torchvision
import os
import random
import pickle
from archs.cifarsvhn.vgg import vgg16_bn, vgg19_bn
from archs.cifarsvhn.resnet import ResNet19
import core
from utils_for_pruning import *
from utils import data_transforms
from spikingjelly.clock_driven.functional import reset_net
from spikingjelly.clock_driven import encoding
from core import Masking
from models import *
import torchvision.transforms as transforms
from data import CIFAR10Policy, Cutout
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


def same_seeds(seed): 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = config.get_args()
    args = parser.parse_args()
    print(args)
    same_seeds(args.seed)
    torch.cuda.set_device(args.gpu)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # define dataset
    train_transform, valid_transform = data_transforms(args)
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        n_class = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        # train_loader, val_loader = build_data(cutout=True, use_cifar10=False, auto_aug=True, batch_size=args.batch_size, args=args)

        n_class =100

    elif args.dataset == "tinyimagenet":
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_dir, 'tiny-imagenet-200/train'), transform=train_transform)
        valset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_dir, 'tiny-imagenet-200/val'), transform=valid_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        n_class = 200
    else:
        print("Wrong Dataset")
        exit()

    criterion = nn.CrossEntropyLoss()

    if  args.arch == 'vgg16':
        model = vgg16_bn(num_classes=n_class)
        model = SpikeModel(model, args.timestep, args)
        model.set_spike_state(True)
    elif args.arch == 'resnet19':
        model = ResNet19(num_classes=n_class)
        model = SpikeModel(model, args.timestep, args)
        model.set_spike_state(True)

    else:
        print("Wrong Arch")
        exit()
    model.cuda()
    print(model)
    
    model.local = 0
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        print ("will be added...")
        exit()

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(140),int(240)], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.end_iter+args.final_prune_epoch), eta_min=0)
    else:
        print ("will be added...")
        exit()
    
    model.total_timestep = args.timestep
    # Making Initial Mask
    mask = Masking(optimizer, prune_rate=args.regeneration_ratio, args=args, train_loader=train_loader)
    mask.add_module(model)

    best_acc = 0
    for epoch in range(args.final_prune_epoch):
        
        train_acc, train_loss = train(args, epoch, train_loader, model, criterion, optimizer, scheduler, mask)
        if(epoch + 1) % 1 == 0:
            accuracy, loss = test(model, val_loader, criterion)
            print('[Epoch:%d] Test Accuracy:%f, Train Loss:%f'
                  % (epoch + 1, 100*accuracy, train_loss))
    print("Pruning Over.")

    train_records = {"loss":[], "acc":[]}
    test_records = {"loss":[], "acc":[]}
    for epoch in range(args.end_iter):
        # break
        acc, loss = train(args, epoch+args.final_prune_epoch, train_loader, model, criterion, optimizer, scheduler, mask)
        train_records["loss"].append(loss)
        train_records["acc"].append(acc)

        if (epoch + 1) % 1 == 0:
            test_accuracy, test_losses = test(model, val_loader, criterion)
            test_records["loss"].append(test_losses)
            test_records["acc"].append(100*test_accuracy)

            print('[Epoch:%d] Test Accuracy:%f, Train Loss:%f'
                  % (epoch + 1, test_accuracy*100, loss))
            if(test_accuracy > best_acc):
                best_acc = test_accuracy
                torch.save(model.state_dict(), os.path.join(args.save, "model_final_best_pruned.pth.tar"))

        # 保存模型和其他训练状态
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }

        # 保存 checkpoint 到文件
        torch.save(checkpoint, os.path.join(args.save, 'full_checkpoint.pth'))


    print("Fine-Tune Over")
    print("Best Accuracy: ", best_acc)
    if(args.show):
        print("Train Loss List: ")
        print(train_records["loss"])
        print("Train Accuracy List: ")
        print(train_records["acc"])
        print("Test Loss List: ")
        print(test_records["loss"])
        print("Test Accuracy List: ")
        print(test_records["acc"])

def train(args, epoch, train_data,  model, criterion, optimizer, scheduler, mask):
    model.train()
    EPS = 1e-6
    losses = []
    accs = []
    
    for batch_idx, (imgs, targets) in enumerate(train_data):
        train_loss = 0.0
        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()
        # output_list, v_list = model(imgs)
        output, v_list = model(imgs)
        out_f = 0
        # for output in output_list:
        train_loss += criterion(output, targets) / args.timestep
        # output = sum(output_list)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(targets.data.view_as(pred)).sum().item()
        acc = correct / targets.size()[0]
        train_loss.backward()
        mask.step(v_list, None)
        # optimizer.step()
        losses.append(train_loss.item())
        accs.append(acc)
        # reset_net(model)
    scheduler.step() 
    return sum(accs)/len(accs), sum(losses)/len(losses)

if __name__ == '__main__':
    main()
