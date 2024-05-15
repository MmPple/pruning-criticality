'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from typing import Union, List, Dict, Any, cast, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from .channel_selection import channel_selection


__all__ = ["ResNet19"]

tau_global = 1./(1. - 0.25)

class newLIFNode(neuron.LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, cupy_fp32_inference=False):
        super().__init__(tau, decay_input, v_threshold,
                 v_reset, surrogate_function,
                 detach_reset, cupy_fp32_inference)
        self.v_keep = 0
    
    def neuronal_charge(self, x: torch.Tensor):
        super().neuronal_charge(x)
        self.v_keep = self.v.detach().clone()

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, pruning=True,v_threshold=1.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.select = channel_selection(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.v_threshold = v_threshold
        self.stride = stride

        self.lif1 = newLIFNode(v_threshold=v_threshold, v_reset=0.0, tau= tau_global,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=False)

        self.lif2 = newLIFNode(v_threshold=v_threshold, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=False)
 
        self.shortcut = nn.Sequential()
        self.isshortcut = False
        if stride != 1 or in_planes != self.expansion*planes:
            self.isshortcut = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        self.pruning = pruning

    def forward(self, x):
        global spike_list
        global v_list
        global time

        out = self.bn1(x)
        out = self.select(out)
        out = self.lif1(out)
        spike_list[time].append(out.detach())
        v_list[time].append(self.lif1.v_keep)

        
        if(self.pruning):
            out = F.relu(self.bn1(self.conv1(x)) - self.v_threshold)
        else:
            if(self.conv1.weight.data.size()[0] == 0):
                out = 0
            else:
                out = self.lif2(self.bn2(self.conv1(out)))
                spike_list[time].append(out.detach())
                v_list[time].append(self.lif2.v_keep)
                out = self.conv2(out)

            
        short_out = self.shortcut(x)

        out += short_out

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)


        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, total_timestep =6, pruning=False, local=False, v_threshold=1.0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.total_timestep = total_timestep
        self.v_threshold = v_threshold
        self.pruning = pruning
        self.local = local
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)


        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
        
        self.bn = nn.BatchNorm2d(512)
        self.select = channel_selection(512)
        self.lif_conv = newLIFNode(v_threshold=v_threshold, v_reset=0.0, tau= tau_global,
                                        surrogate_function=surrogate.ATan(),
                                        detach_reset=True)

        self.fc1 = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.pruning, self.v_threshold))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        output_list = []
        global spike_list 
        spike_list = [[] for i in range(self.total_timestep)]
        global v_list 
        v_list = [[] for i in range(self.total_timestep)]
        global time
        time = 0
        static_x = self.conv1(x)

        for t in range(self.total_timestep):
            time = t
            
            out = self.layer1(static_x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.bn(out)
            out = self.select(out)
            out = self.lif_conv(out)

            spike_list[t].append(out.detach())
            v_list[t].append(self.lif_conv.v_keep)
            
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)

            out = self.fc1(out)
            output_list.append(out)

        return output_list, (spike_list, v_list)


def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet19(num_classes, total_timestep, pruning=False, local=0, v_threshold=1.0):
    return ResNet(BasicBlock, [3,3,2], num_classes, total_timestep, pruning, local, v_threshold)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()


