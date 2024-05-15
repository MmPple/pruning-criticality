'''
Modified based on the code of Sparse Training via Boosting Pruning Plasticity with Neuroregeneration
'''
from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import math


class Masking(object):
    def __init__(self, optimizer, prune_rate=0.3, growth_death_ratio=1.0, prune_rate_decay=None, death_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, args=None, train_loader=None, device=None):

        self.args = args
        self.loader = train_loader
        self.device = device
        self.args.final_density = 1 - self.args.final_sparsity

        self.masks = {}
        self.l1masks = {}
        self.lastmasks = {}
        self.react_nums = []
        self.final_masks = {}
        self.grads = {}
        self.nonzero_masks = {}
        self.scores = {}
        self.pruning_rate = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.total_params = 0
        self.fc_params = 0
        self.prune_rate = prune_rate
        self.name2prune_rate = {}
        self.steps = 0
        self.grads = []
        self.curr_prune_iter = 0

        self.prune_every_k_steps = self.args.update_frequency


    def init(self, mode='ER', density=0.05, erk_power_scale=1.0, grad_dict=None):
        print('initialize by ERK')
        for name, weight in self.masks.items():
            self.total_params += weight.numel()
            if 'classifier' in name:
                self.fc_params = weight.numel()
        is_epsilon_valid = False
        dense_layers = set()
        while not is_epsilon_valid:

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                    rhs -= n_zeros

                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (
                                                        np.sum(mask.shape) / np.prod(mask.shape)
                                                ) ** erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            epsilon = rhs / divisor
            # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        density_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in self.masks.items():
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                density_dict[name] = probability_one
            self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda(self.device)

            total_nonzero += density_dict[name] * mask.numel()
        print(f"Overall sparsity {total_nonzero / self.total_params}")
        self.apply_mask()

    def step(self, v_list, writer):
        self.optimizer.step()
        self.apply_mask()
        self.steps += 1
        if self.prune_every_k_steps is not None:
                self.pruning(self.steps, None, v_list)
                self.grads.clear()

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}, density: {2:.3f}'.format(name, num_nonzeros, num_nonzeros/float(mask.numel()))
                print(val)

        print('Death rate: {0}\n'.format(self.prune_rate))

    def pruning(self, step, spikes, v_list):

        ini_time = int(self.args.init_prune_epoch * len(self.loader))
        final_time = int(self.args.final_prune_epoch * len(self.loader))
        curr_prune_iter = self.curr_prune_iter + 1
        final_iter = int((self.args.final_prune_epoch * len(self.loader)) / self.prune_every_k_steps)
        ini_iter = int((self.args.init_prune_epoch * len(self.loader)) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter

        if(step < ini_time or step % self.prune_every_k_steps != 0): return
        self.curr_prune_iter += 1

        # calculate criticality scores
        if(len(self.grads)==0):
            self.grads = [[], []]
        for l, (v, s) in enumerate(v_list):
            temp = 3.0
            # out_bp = torch.clamp(v, 0, 1)
            out_bp = v
            fa = (torch.tanh(temp * (out_bp-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
            # fa =  temp  / ((2 * (np.tanh(temp * 0.5))) * torch.cosh(temp * (out_bp-0.5)) * torch.cosh(temp * (out_bp-0.5)))
            # fa = 4 * torch.sigmoid(4*(out_bp-0.5)) * (1-torch.sigmoid(4*(out_bp-0.5))) 
            # fa = 1 / (1+((math.pi * (out_bp-0.5))**2))
            # print(fa.size())
            s = s.mean(axis=[0,1])
            fa = fa.mean(axis=[0])
            if(len(fa.size()) > 2):
                fa = fa.flatten(2).max(dim=2)[0]
            fa = fa.mean(axis=[0])
            # fa = (1000 / fa.size()) + fa
            # print(fa.size())
            # print(fa.mean())
            if(len(self.grads[0]) < len(v_list)):
                self.grads[1].append(fa)
                self.grads[0].append(s)
            else:
                self.grads[1][l] += fa
                self.grads[0][l] += s
        # (spike_list,v_list) = v_list
        # fa_list = []
        # s_list = []
        # for t, vt in enumerate(v_list):
        #     for l, v in enumerate(vt):
        #         v = v - 1.0
        #         fa = 2.0 / 2 / (1 + (math.pi / 2 * 2.0 * v).pow_(2))
        #         if(t==0):
        #             fa_list.append(fa)
        #             s_list.append(spike_list[t][l])
        #         else:
        #             fa_list[l] = fa_list[l] + fa
        #             s_list[l] += spike_list[t][l]
        #         if(t== (len(v_list)-1)):
        #             fa_list[l] = fa_list[l] / len(v_list)
        #             s_list[l] = s_list[l] / len(v_list)
        #             if(len(fa_list[l].size()) > 2):
        #                 fa_list[l] = fa_list[l].flatten(2).max(dim=2)[0]
        #             if(len(self.grads[1]) <  len(vt)):
        #                 self.grads[1].append(fa_list[l].mean(axis = [0]))
        #                 self.grads[0].append(s_list[l].mean(axis = [0]))
        #             else:
        #                 self.grads[1][l] = self.grads[1][l] + fa_list[l].mean(axis = [0])
        #                 self.grads[0][l] = self.grads[0][l] + s_list[l].mean(axis = [0])


        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter:
            print('******************************************************')
            print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
            print('******************************************************')
            prune_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.args.final_density) * (
                    1 - prune_decay)
            
            weight_abs = []
            for module in self.modules:
                cnt = 0
                for name, weight in module.named_parameters():  
                    if name not in self.masks: continue
                    if(len(weight.shape) == 2):    
                        weight_abs.append(weight.abs())
                        cnt += 1
                        continue 
                    weight_abs.append(weight.abs())
                    cnt += 1

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1 - curr_prune_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]
            for module in self.modules:
                cnt = 0
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.l1masks[name] = ((weight_abs[cnt]) > acceptable_score).float() # must be > to prevent acceptable_score is zero, leading to dense tensors
                    cnt += 1
            num_lastlayer_keep = (weight_abs[-1] > acceptable_score).sum().item()
            num_params_to_keep1 = num_params_to_keep - num_lastlayer_keep
            self.growth_num = int(num_params_to_keep1 * self.prune_rate)
            num_params_to_keep1 = num_params_to_keep1 - self.growth_num

            all_scores = torch.cat([torch.flatten(x) for x in weight_abs[:-1]])
            threshold, _ = torch.topk(all_scores, num_params_to_keep1, sorted=True)
            if(self.prune_rate == 1.0):
                acceptable_score1 = 1e6
            else:
                acceptable_score1 = threshold[-1]

            for module in self.modules:
                cnt = 0
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    if(cnt == (len(weight_abs)-1)):
                        self.masks[name] = ((weight_abs[cnt]) > acceptable_score).float() # must be > to prevent acceptable_score is zero, leading to dense tensors
                    else:
                        self.masks[name] = ((weight_abs[cnt]) > acceptable_score1).float() # must be > to prevent acceptable_score is zero, leading to dense tensors
                    cnt += 1 

            self.apply_mask()
            # self.print_nonzero_counts()
            self.truncate_weights(self.steps)
            print("growth...")
            # self.print_nonzero_counts()

            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print('Total Model parameters:', total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()
            print('Now pruned Model parameters:', sparse_size)
            print('Sparsity after pruning: {}'.format(
                (total_size-sparse_size) / total_size))

            num_react = 0
            total_prune = 0

            for module in self.modules:
                cnt = 0
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mnow = (self.l1masks[name]==0) * (self.lastmasks[name]==1)
                    preact = (mnow * (self.masks[name] == 1)).sum().item()
                    total_prune += mnow.sum().item()
                    num_react += preact
                    self.lastmasks[name] = self.masks[name]
                    cnt += 1
            self.react_nums.append(num_react / total_prune)
            if(self.args.proportion):
                print("Regeneration Survival Percentage from Pruning: {:.2f}%".format(num_react/total_prune*100))

    def add_module(self, module, sparse_init='ERK', init_density=1.0, grad_dic=None):
        self.module = module
        self.sparse_init = sparse_init
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            if len(tensor.size()) == 4 or len(tensor.size()) == 2:
                self.names.append(name)
                self.masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda(self.device)
                self.l1masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda(self.device)
                self.lastmasks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda(self.device)

        self.init(mode=sparse_init, density=init_density, grad_dict=grad_dic)



    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data*self.masks[name]
    
    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]

                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

    def truncate_weights(self, step=None, spikes=None):
        if(self.prune_rate == 0):
            return
                    
        total_notzeros = 0
        total_zeros = 0
        weight_list = []
        ff_list = []
        cnt = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if(cnt == len(self.grads[1])):
                    break
                elif(len(self.masks[name].size()) == 4):
                    ff =  (self.grads[1][cnt].view(-1,1,1,1))
                    # ff = ff / ff.sum() 
                    ff = ff * (self.masks[name]==0)
                    # ff =  (torch.rand(self.grads[1][cnt].shape).view(-1,1,1,1).cuda()) * (self.masks[name]==0)
                    # ff =  (0-self.grads[1][cnt].view(-1,1,1,1)) * (self.masks[name]==0)
                    # ff =  weight.grad.clone().abs() * (self.masks[name]==0)
                elif(len(self.masks[name].size()) == 2):
                    # ff = (self.grads[1][cnt].view(-1,1,1,1) * self.grads[0][cnt-1].view(1,-1,1,1)) * (self.masks[name]==0)
                    ff =  (self.grads[1][cnt].view(-1,1)) 
                    # ff = ff / ff.sum() 
                    ff = ff * (self.masks[name]==0)
                    # ff =  (torch.rand(self.grads[1][cnt].shape).view(-1,1).cuda()) * (self.masks[name]==0)
                    # ff =  (0-self.grads[1][cnt].view(-1,1)) * (self.masks[name]==0)
                    # ff = weight.grad.clone().abs() * (self.masks[name]==0)

                ff_list.append(ff)
                # print(ff.sum() / ((ff!=0).sum()+1e-8))
                cnt += 1
            
            all_scores = torch.cat([torch.flatten(x) for x in ff_list])
            threshold, _ = torch.topk(all_scores, self.growth_num, sorted=True)
            acceptable_score = threshold[-1]
            cnt = 0 
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if(cnt == len(ff_list)): break
                self.masks[name] = self.masks[name] + (ff_list[cnt] >= acceptable_score).float()
                cnt += 1
        
        self.apply_mask()
