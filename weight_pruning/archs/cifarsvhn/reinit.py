import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
import pickle
import math


def kaiming_uniform_(mask, tensor):

    a = math.sqrt(5)
    gain = math.sqrt(2.0 / (1 + a ** 2))

    for i, m in enumerate(mask):
        fan = m.sum()
        if(fan == 0): continue
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            tmp_vec = torch.zeros(int(fan))
            tmp_vec.uniform_(-bound, bound)
            for e, index in zip(tmp_vec, torch.nonzero(torch.from_numpy(m))):
                if len(index) == 1:
                    tensor[i, index[0]] = e
                elif len(index) == 3:
                    tensor[i, index[0], index[1], index[2]] = e
                else:
                    print("Wrong Mask !")

def init(mask, weight, name):
    # print(weight.size())
    if (len(weight.shape)) != 1:
        print(f"reinit {name}")
        kaiming_uniform_(mask, weight)
    else:
        pass

