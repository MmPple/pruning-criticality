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

def same_seeds(seed): 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main():
    parser = config.get_args()
    args = parser.parse_args()
    print(args)
    same_seeds(args.seed)
    torch.cuda.set_device(args.gpu)

    if not os.path.exists(args.save):
        print("No Model !")
        exit()

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
        
    model.total_timestep = args.timestep
    state = torch.load(os.path.join(args.save, "model_final_best_pruned.pth.tar"))
    model.load_state_dict(state)


    train_means = extraction(args, train_loader, model, n_class)
    test_means = extraction(args, val_loader, model, n_class)
    print("Intra-cluster variance of classes for Train set: ")
    print([s[1] for s in train_means])
    print("Intra-cluster variance of classes for Test set: ")
    print([s[1] for s in test_means])
    cs = []
    for i, (s1, s2) in enumerate(zip(train_means, test_means)):  
        cs.append(torch.cosine_similarity(s1[0], s2[0], dim=0).item())
    print("Cosine Similarity for classes between trainset and testset:")
    print(cs)


def extraction(args, data_loader,  model, n_class):
    model.eval()

    s_map = [[] for i in range(n_class)]
    for batch_idx, (imgs, targets) in enumerate(data_loader):
        imgs, targets = imgs.cuda(), targets.cuda()
        output_list, v_list = model(imgs)
        # spike_list = [[] for i in range(model.step)]
            
        #     for t in range(len(spike_list)):
        #         spike_list[t].append(s[t])
        s_list = []
        for l, (v, s) in enumerate(v_list):
            s = s.detach()
            s_list.append(s.sum(dim=0))
        # for t in range(len(spike_list)):
        #     for l in range(len(spike_list[t])):
        #         if(t==0):
        #             s_list.append(spike_list[t][l])
        #         else:
        #             s_list[l] += spike_list[t][l]
        for l in range(len(s_list)):
            if l < (len(s_list) -1): continue
            if(len(s_list[l].size()) > 2):
                # if(l == 2):
                # print(s_list[l].sum(axis=[2,3])[indexs])
                s_t = s_list[l].sum(axis=[2,3])
                s_t = (s_t.max(dim=1, keepdim=True)[0] - s_t) / (s_t.max(dim=1, keepdim=True)[0]-s_t.min(dim=1, keepdim=True)[0])
                for i, k in enumerate(targets):
                    index = k.item()
                    if len(s_map[index]) == 0:
                        s_map[index].append(s_t[[i]])
                    else:
                        s_map[index][0] = torch.cat([s_map[index][0], s_t[[i]]], dim=0)

    means = []
    for index, ss in enumerate(s_map):
        means.append([0,0])
        for i, s in enumerate(ss):
            v = s.var(dim=0)
            s = s.mean(dim=0)
            means[index][1] = v.mean().item()
            means[index][0] = s

    return means
    # return

if __name__ == '__main__':
    main()
