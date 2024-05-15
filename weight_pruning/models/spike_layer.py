import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from IPython import embed


class myRN(nn.Module):
    def __init__(self, feature_channels, eps=1e-05, momentum=0.1):
        super(myRN, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
               condition Mask: (B,1,H,W): 0 for background, 1 for foreground
        return: tensor RN_B(x): (N,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''

        # gamma and beta
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(feature_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        
        self.register_buffer('running_mean', torch.zeros(feature_channels))
        self.register_buffer('running_var', torch.ones(feature_channels))

        
        # RN
        # self.bn_norm = nn.BatchNorm2d(feature_channels, affine=False, track_running_stats=False)


    def forward(self, x, mask):
        if mask.size(0) != x.size(1):
            mask = torch.ones(x.size()[1:]).cuda(x.device)
        mask = mask.detach()
        region = x * mask
        if self.training:      
            shape = region.size()
            n = x.size(0)
            sum = torch.sum(region, dim=[0,2,3])  # (B, C) -> (C)
            Sr = torch.sum(mask, dim=[1,2]) * n   # (B, 1) -> (1)
            Sr[Sr==0] = 1
            mean = (sum / Sr)
            var = (region + (1-mask) * mean[:,None,None]).var(axis = [0,2,3], unbiased=False)
            var =  var * n * mask.size(1) * mask.size(2) / Sr
            Sr_ = Sr-1
            Sr_[Sr_==0] = 1
            # Update running mean and variance (for training only)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * Sr / Sr_

            # Element-wise batch normalization
            x_normalized = (region - mean[:,None,None]*mask) / torch.sqrt(var[:,None,None] + self.eps)
        else:
            # Use running mean and variance for inference
            x_normalized = (region - self.running_mean[:,None,None]*mask) / torch.sqrt(self.running_var + self.eps)[:,None,None]
 
        # Scale and shift
        x_normalized = x_normalized * self.weight[None,:,None,None]
        x_normalized = x_normalized + self.bias[None,:,None,None]*mask

        return x_normalized


    def rn(self, region, mask):
        '''
        input:  region: (B,C,M,N), 0 for surroundings
                mask: (B,1,M,N), 1 for target region, 0 for surroundings
        output: rn_region: (B,C,M,N)
        '''
        shape = region.size()
        sum = torch.sum(region, dim=[0,2,3])  # (B, C) -> (C)
        Sr = torch.sum(mask.unsqueeze(0), dim=[0,2,3])    # (B, 1) -> (1)
        Sr[Sr==0] = 1
        mu = (sum / Sr)     # (B, C) -> (C)
        return self.bn_norm(region + (1 - mask.unsqueeze(0)) * mu[None,:,None,None])
        
        # return self.bn_norm(region + (1 - mask) * mu[None,:,None,None]) * \
        # (torch.sqrt(Sr / (shape[0] * shape[2] * shape[3])))[None,:,None,None]


class RN(nn.Module):
    def __init__(self, feature_channels):
        super(RN, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
               condition Mask: (B,1,H,W): 0 for background, 1 for foreground
        return: tensor RN_B(x): (N,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''

        # gamma and beta
        self.weight = nn.Parameter(torch.ones(feature_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        # RN
        self.bn_norm = nn.BatchNorm2d(feature_channels, affine=False, track_running_stats=False)
        

    def forward(self, x, mask):

        mask = mask.detach()

        rn_x = self.rn(x * mask, mask)

        rn_x = rn_x * self.weight[None,:,None,None] + self.bias[None,:,None,None]

        return rn_x


    def rn(self, region, mask):
        '''
        input:  region: (B,C,M,N), 0 for surroundings
                mask: (B,1,M,N), 1 for target region, 0 for surroundings
        output: rn_region: (B,C,M,N)
        '''
        shape = region.size()
        sum = torch.sum(region, dim=[0,2,3])  # (B, C) -> (C)
        Sr = torch.sum(mask.unsqueeze(0), dim=[0,2,3])    # (B, 1) -> (1)
        Sr[Sr==0] = 1
        mu = (sum / Sr)     # (B, C) -> (C)
        return self.bn_norm(region + (1 - mask.unsqueeze(0)) * mu[None,:,None,None])
        
        # return self.bn_norm(region + (1 - mask) * mu[None,:,None,None]) * \
        # (torch.sqrt(Sr / (shape[0] * shape[2] * shape[3])))[None,:,None,None]

class ElementBN(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(ElementBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable scale and shift parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Statistics for inference
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    def forward(self, x, mask=None):
        if self.training:
            # Calculate mean and variance along each feature dimension for the current batch
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)
            n = x.size(0)
            # Update running mean and variance (for training only)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze(0)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze(0) * n / (n-1)

            # Element-wise batch normalization
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        else:
            # Use running mean and variance for inference
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # Scale and shift
        x_normalized = x_normalized * self.weight.unsqueeze(0)
        x_normalized = x_normalized + self.bias.unsqueeze(0) 

        return x_normalized

class SpikeModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._spiking = False

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike

    def forward(self, x):
        # shape correction
        if self._spiking is not True and len(x.shape) == 5:
            x = x.mean([0])
        return x


def spike_activation(x, ste=False, temp=1.0): 
    out_s = torch.gt(x, 0.5)
    if ste:
        out_bp = torch.clamp(x, 0, 1)
    else:
        out_bp = torch.clamp(x, 0, 1)
        out_bp = (torch.tanh(temp * (out_bp-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
        # out_bp = torch.sigmoid(4*(out_bp-0.5))
        # out_bp = (1/math.pi) * torch.atan(math.pi * (out_bp-0.5)) + 1/2
    return (out_s.float() - out_bp).detach() + out_bp


def MPR(s,thresh):

    s[s>1.] = s[s>1.]**(1.0/3)
    s[s<0.] = -(-(s[s<0.]-1.))**(1.0/3)+1.
    s[(0.<s)&(s<1.)] = 0.5*torch.tanh(3.*(s[(0.<s)&(s<1.)]-thresh))/np.tanh(3.*(thresh))+0.5
    
    return s


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


def mem_update(mask, bn, x_in, mem, V_th, decay, grad_scale=1., temp=1.0):
    # x_in = bn(x_in, mask)
    mem = mem * decay + x_in
    #if mem.shape[1]==256:
    #    embed()
    #V_th = gradient_scale(V_th, grad_scale)
    #mem2 = MPR(mem, 0.5)
    
    # mem2 = bn(mem, mask)
    # mem2 = bn(mem)
    mem2 = mem
    spike = spike_activation(mem2/V_th, temp=temp)
    mem = mem * (1 - spike)
    #mem = mem - spike
    #spike = spike * Fire_ratio
    return mem, mem2, spike




class LIFAct(SpikeModule):
    """ Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, step, channel=None):
        super(LIFAct, self).__init__()
        self.step = step
        #self.V_th = nn.Parameter(torch.tensor(1.))
        self.V_th = 1.0
        # self.tau = nn.Parameter(torch.tensor(-1.1))
        self.temp = 3.0
        #self.temp = nn.Parameter(torch.tensor(1.))
        self.grad_scale = 0.1
        
        # self.bn = myRN(channel[0])
        self.bn = None
        # self.bn = RN(channel[0])
        # self.bn = ElementBN(channel)
        # self.bn = nn.BatchNorm2d(channel[0])
        
        #nn.init.constant_(self.bn.weight, 1)
        #nn.init.constant_(self.bn.bias, 0.5)

    def forward(self, x, mask=None):
        if self._spiking is not True:
            return F.relu(x)
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel()*self.step)
        u = torch.zeros_like(x[0])
        out = []
        v = []
        for i in range(self.step):
        
            u, ut, out_i = mem_update(mask = mask, bn=self.bn, x_in=x[i], mem=u, V_th=self.V_th,
                                  grad_scale=self.grad_scale, decay=0.25, temp=self.temp)
            # out_i = out_i * mask.unsqueeze(0).detach()
            # out_i = out_i * mask.unsqueeze(0)
            out += [out_i]
            
            v += [ut.detach()]

        out = torch.stack(out)
        v = torch.stack(v)
        # print(out.shape, out.sum())
        return out, v



class SpikeConv(SpikeModule):


    def __init__(self, conv, step=2):
        super(SpikeConv, self).__init__()
        self.conv = conv
        self.step = step

    def forward(self, x):
        if self._spiking is not True:
            return self.conv(x)
        out = []
        for i in range(self.step):
            out += [self.conv(x[i])]
        out = torch.stack(out)
        return out


class SpikePool(SpikeModule):

    def __init__(self, pool, step=2):
        super().__init__()
        self.pool = pool
        self.step = step

    def forward(self, x):
        if self._spiking is not True:
            return self.pool(x)
        T, B, C, H, W = x.shape
        out = x.reshape(-1, C, H, W)
        out = self.pool(out)
        B_o, C_o, H_o, W_o = out.shape
        out = out.view(T, B, C_o, H_o, W_o).contiguous()
        return out

class myBatchNorm3d(SpikeModule):
    def __init__(self, BN: nn.BatchNorm2d, step=2):
        super().__init__()
        self.bn = nn.BatchNorm3d(BN.num_features)
        self.step = step
    def forward(self, x):
        if self._spiking is not True:
            return BN(x)
        out = x.permute(1, 2, 0, 3, 4)
        out = self.bn(out)
        out = out.permute(2, 0, 1, 3, 4).contiguous()
        return out

class myRN3d(SpikeModule):
    def __init__(self, num_features, step=2):
        super().__init__()
        self.bn = myRN(num_features)
        self.step = step
    def forward(self, x, mask=None):
        out = []
        # print(x.size(), mask.size())            
        for i in range(self.step):
            out_i = self.bn(x[i], mask)
            out += [out_i]
        out = torch.stack(out)
        return out



class myNone(SpikeModule):
    def __init__(self, step=2):
        super().__init__()

    def forward(self, x):
        out = x
        return out




class tdBatchNorm2d(nn.BatchNorm2d, SpikeModule):
    """Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """

    def __init__(self, bn: nn.BatchNorm2d, alpha: float):
        super(tdBatchNorm2d, self).__init__(bn.num_features, bn.eps, bn.momentum, bn.affine, bn.track_running_stats)
        self.alpha = alpha
        self.V_th = 0.5
        # self.weight.data = bn.weight.data
        # self.bias.data = bn.bias.data
        # self.running_mean.data = bn.running_mean.data
        # self.running_var.data = bn.running_var.data

    def forward(self, input):
        if self._spiking is not True:
            # compulsory eval mode for normal bn
            self.training = False
            return super().forward(input)

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 1, 3, 4])
            # use biased var in train
            var = input.var([0, 1, 3, 4], unbiased=False)
            n = input.numel() / input.size(2)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        channel_dim = input.shape[2]
        input = self.alpha * self.V_th * (input - mean.reshape(1, 1, channel_dim, 1, 1)) / \
                (torch.sqrt(var.reshape(1, 1, channel_dim, 1, 1) + self.eps))
        if self.affine:
            input = input * self.weight.reshape(1, 1, channel_dim, 1, 1) + self.bias.reshape(1, 1, channel_dim, 1, 1)

        return input
