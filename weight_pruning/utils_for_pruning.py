import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from archs.cifarsvhn.reinit import *
from spikingjelly.clock_driven.functional import reset_net



def dfs_net(model, mask, cnt_list):
    for name, m in model.named_modules():
        lid = get_id(name)
        if lid == 0:
            pass
        elif lid == 4:
            cnt_list[2] = cnt_list[2] + 1 
        elif lid == 1:
            tensor = torch.from_numpy(mask[cnt_list[2]]).cuda()
            cnt_list[0].append( \
            (tensor.sum(axis=[1,2,3]) if len(tensor.shape) == 4 else tensor.sum(axis=1) ))
            cnt_list[1].append( \
            (tensor.shape[1]*tensor.shape[2]*tensor.shape[3] if len(tensor.shape) == 4 else tensor.shape[1]))
            cnt_list[2] = cnt_list[2] + 1
        elif lid == 2:
            m.v_threshold = (cnt_list[0][cnt_list[3]] * 1.0 / cnt_list[1][cnt_list[3]]) * 1.0
            
            if(not "lif_fc" in name):
                m.v_threshold = m.v_threshold.view(-1, 1, 1)
            # m.v_threshold = 0.8
            cnt_list[3] += 1
        elif lid == 3:
            cnt_list[2] = cnt_list[2] + 1


def get_id(name):
    if 'conv' in name or ('fc' in name and 'lif' not in name):
        return 1
    elif 'lif' in name and 'surrogate_function' not in name:
        return 2
    elif 'bn' in name or 'shortcut.1' in name:
        return 3
    elif 'shortcut.0' in name:
        return 4 
    elif 'pool' in name:
        return 0
    else:
        return 0


def reset_threshold(model, mask, v_base=0.0):
    step = 0
    cnt_list = [[], [], 0, 0]
    dfs_net(model, mask, cnt_list)
    print(model)
    


def make_mask(model):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

    return mask

# Prune by Percentile module
def prune_by_percentile(args, percent, mask , model):


        if args.pruning_scope == 'local':
            # Calculate percentile value
            step = 0
            for name, param in model.named_parameters():

                # We do not prune bias term
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    if (len(tensor.shape)) == 1:
                        step += 1
                        continue
                    alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                    percentile_value = np.percentile(abs(alive), percent)

                    # Convert Tensors to numpy and calculate
                    weight_dev = param.device
                    new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                    # Apply new weight and mask
                    param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                    mask[step] = new_mask
                    step += 1
            step = 0
        elif args.pruning_scope == 'global':
            step = 0
            all_param = []
            for name, param in model.named_parameters():
                # We do not prune bias term
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    if (len(tensor.shape)) == 1: # We do not prune BN term
                        continue
                    alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
                    all_param.append(list(abs(alive)))
            param_whole = np.concatenate(all_param)
            percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0])/float(100./percent))]

            step = 0

            for name, param in model.named_parameters():
                # We do not prune bias term
                if 'weight' in name:
                    tensor =  param.data.cpu().numpy()
                    if (len(tensor.shape)) == 1:  # We do not prune BN term
                        step += 1
                        continue

                    # Convert Tensors to numpy and calculate
                    weight_dev = param.device
                    new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                    # Apply new weight and mask
                    param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                    mask[step] = new_mask
                    step += 1
            step = 0
        else:
            exit()

        return model, mask


def get_pruning_maks(args, percent, mask, model):
    step = 0
    all_param = []
    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            if (len(tensor.shape)) == 1:  # We do not prune BN term
                continue
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            all_param.append(list(abs(alive)))
    param_whole = np.concatenate(all_param)
    percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0]) / float(100. / percent))]

    step = 0

    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            if (len(tensor.shape)) == 1:  # We do not prune BN term
                step += 1
                continue
            new_mask = np.where(abs(tensor) < percentile_value, 0, torch.FloatTensor([1]))
            mask[step] = new_mask
            step += 1
    step = 0

    return  mask


def original_initialization(mask_temp, initial_state_dict, model):

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

    return model
# def make_mask(model):
#     step = 0
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             step = step + 1
#     mask = [None]* step
#     step = 0
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             tensor = param.data.cpu().numpy()
#             mask[step] = np.ones_like(tensor)
#             step = step + 1
#     step = 0

#     return mask

# def _normalize_scores(scores):
#     """
#     Normalizing scheme for LAMP.
#     """
#     # sort scores in an ascending order
#     sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
#     # compute cumulative sum
#     scores_cumsum_temp = sorted_scores.cumsum(dim=0)
#     scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
#     scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
#     # normalize by cumulative sum
#     sorted_scores /= (scores.sum() - scores_cumsum)
#     # tidy up and output
#     new_scores = torch.zeros(scores_cumsum.shape,device=scores.device)
#     new_scores[sorted_idx] = sorted_scores
    
#     return new_scores.view(scores.shape)



# # Prune by Percentile module
# def prune_by_percentile(args, percent, mask , model):


#         if args.pruning_scope == 'local':
#             # Calculate percentile value
#             step = 0
#             for name, param in model.named_parameters():

#                 # We do not prune bias term
#                 if 'weight' in name:
#                     tensor = param.data.cpu().numpy()
#                     if (len(tensor.shape)) == 1:
#                         step += 1
#                         continue
#                     alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
#                     percentile_value = np.percentile(abs(alive), percent)

#                     # Convert Tensors to numpy and calculate
#                     weight_dev = param.device
#                     new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

#                     # Apply new weight and mask
#                     param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
#                     mask[step] = new_mask
#                     step += 1
#             step = 0
#         elif args.pruning_scope == 'global':
#             step = 0
#             all_param = []
#             for name, param in model.named_parameters():
#                 # We do not prune bias term
#                 if 'weight' in name:
#                     tensor = param.data.cpu().numpy()
#                     if (len(tensor.shape)) == 1: # We do not prune BN term
#                         continue

#                     # tensor = _normalize_scores(param.data**2).cpu().numpy()
#                     alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
#                     all_param.append(list(abs(alive)))
#             param_whole = np.concatenate(all_param)
#             percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0])/float(100./percent))]

#             step = 0

#             for name, param in model.named_parameters():
#                 # We do not prune bias term
#                 if 'weight' in name:
#                     tensor =  param.data.cpu().numpy()
#                     if (len(tensor.shape)) == 1:  # We do not prune BN term
#                         step += 1
#                         continue

#                     # Convert Tensors to numpy and calculate
#                     weight_dev = param.device
#                     # tensor_ = _normalize_scores(param.data**2).cpu().numpy()
#                     new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
#                     # new_mask = np.where(abs(tensor_) < percentile_value, 0, mask[step])

#                     # Apply new weight and mask
#                     param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
#                     mask[step] = new_mask
#                     step += 1
#             step = 0
#         else:
#             exit()

#         return model, mask

# def prune_by_percentile_s(args, percent, mask , model, spiking_list = None):


#         if args.pruning_scope == 'local':
#             # Calculate percentile value
#             step = 0
#             for name, param in model.named_parameters():

#                 # We do not prune bias term
#                 if 'weight' in name:
#                     tensor = param.data.cpu().numpy()
#                     if (len(tensor.shape)) == 1:
#                         step += 1
#                         continue
#                     alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
#                     percentile_value = np.percentile(abs(alive), percent)

#                     # Convert Tensors to numpy and calculate
#                     weight_dev = param.device
#                     new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

#                     # Apply new weight and mask
#                     param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
#                     mask[step] = new_mask
#                     step += 1
#             step = 0
#         elif args.pruning_scope == 'global':
#             step = 0
#             cnt = 0
#             all_param = []
#             for name, param in model.named_parameters():
#                 # We do not prune bias term
#                 if 'weight' in name:
#                     tensor = param.data.cpu().numpy()
#                     if (len(tensor.shape)) == 1: # We do not prune BN term
#                         continue
#                     tensor = _normalize_scores((param.data * spiking_list[cnt]) ** 2).cpu().numpy()
#                     cnt += 1
#                     alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
#                     # all_param.append(list(abs(alive)))
#                     all_param.append(list(alive))
#             param_whole = np.concatenate(all_param)
#             percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0])/float(100./percent))]

#             step = 0
#             cnt = 0 

#             for name, param in model.named_parameters():
#                 # We do not prune bias term
#                 if 'weight' in name:
#                     tensor =  param.data.cpu().numpy()
#                     if (len(tensor.shape)) == 1:  # We do not prune BN term
#                         step += 1
#                         continue

#                     # Convert Tensors to numpy and calculate
#                     weight_dev = param.device
#                     tensor_ =  _normalize_scores((param.data * spiking_list[cnt]) ** 2).cpu().numpy()
#                     cnt += 1
#                     # new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
#                     new_mask = np.where(tensor_ < percentile_value, 0, mask[step])

#                     # Apply new weight and mask
#                     param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
#                     mask[step] = new_mask
#                     step += 1
#             step = 0
#         else:
#             exit()

#         return model, mask


# def get_pruning_maks_with_grad(args, percent, mask, model, epoch, grads):
#     step = 0
#     cnt = 0
#     all_param = []
#     for name, param in model.named_parameters():
#         # print(name)
#         # We do not prune bias term
#         if 'weight' in name:
#             if (len(param.data.shape)) == 1:  # We do not prune BN term
#                 # print("SKIP:  " + name)
#                 continue
#             tensor = param.data.cpu().numpy()
#             if(epoch > 0):
#                 tensor = (param.data * grads[cnt]).cpu().numpy()
#                 cnt += 1
#             alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
#             all_param.append(list(abs(alive)))
#             # alive = _normalize_scores(torch.from_numpy(alive)**2).numpy() 
#             # all_param.append(list(alive))

#     param_whole = np.concatenate(all_param)
#     percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0]) / float(100. / percent))]

#     step = 0
#     cnt = 0

#     for name, param in model.named_parameters():
#         # We do not prune bias term
#         if 'weight' in name:
#             tensor = param.data.cpu().numpy()
#             if (len(tensor.shape)) == 1:  # We do not prune BN term
#                 step += 1
#                 continue
#             if(epoch > 0):
#                 tensor = (param.data * grads[cnt]).cpu().numpy()
#                 cnt += 1
#             new_mask = np.where(abs(tensor) < percentile_value, 0, torch.FloatTensor([1]))
#             # new_mask = np.where(_normalize_scores(torch.from_numpy(tensor)**2).numpy() < percentile_value, 0, torch.FloatTensor([1]))
#             mask[step] = new_mask
#             step += 1
#     step = 0

#     return  mask


# def get_pruning_maks_with_arch(args, percent, mask, model,epoch, spiking_list=None):
#     step = 0
#     cnt = 0
#     all_param = [] 
#     percentile_value = []
#     # arch = [674, 29340, 59749, 127951, 256058, 543649, 551459, 1139456, 2336598, 2344919, 2349343, 2354464, 2355082, 4639]
#     # arch = [674, 29340, 59749, 127951, 256058, 543649, 551459, 1139456, 2336598, 2344919, 2349343, 2354464, 2355082, 4639]
#     # arch = [357, 19572, 43409, 96618, 198469, 448448, 460368, 989905, 2160663, 2215052, 2204743, 2204463, 2196442, 3880]
#     # arch = [357, 18937, 38601, 88005, 180326, 417798, 439562, 974781, 2169018, 2222034, 2220167, 2228895, 2239752, 4129]
#     # arch = [311, 56819, 118456, 3599, 115606, 121420, 131790, 136225, 273993, 573593, 20657, 581153, 585505, 589542, 589616, 1176574, 2356841, 109423, 2359053, 2359236, 123483, 1079]
#     # arch = [145, 37261, 76444, 2045, 72068, 73710, 79305, 86912, 186150, 431992, 11565, 443043, 485535, 551044, 563235, 1102191, 2274794, 69699, 2353608, 2355798, 89477, 592]
#     arch = [287, 38641, 80270, 3388, 76830, 79267, 83140, 88817, 212703, 453454, 18479, 442885, 479581, 509885, 525950, 1131443, 2309549, 98560, 2307140, 2315267, 99132, 1877]

#     for name, param in model.named_parameters():
#         # print(name)
#         # We do not prune bias term
#         if 'weight' in name:
#             if (len(param.data.shape)) == 1:  # We do not prune BN term
#                 # print("SKIP:  " + name)
#                 continue
#             tensor = param.data.cpu().numpy()
#             # if(epoch > 0):
#             #     tensor = (param.data * spiking_list[cnt]).cpu().numpy()
#             alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
#             alive = abs(alive)
#             percentile_value.append(np.sort(alive)[arch[cnt]])
#             # alive = _normalize_scores(torch.from_numpy(alive)**2).numpy() 
#             # all_param.append(list(alive))
#             cnt += 1

#     step = 0
#     cnt = 0

#     for name, param in model.named_parameters():
#         # We do not prune bias term
#         if 'weight' in name:
#             tensor = param.data.cpu().numpy()
#             if (len(tensor.shape)) == 1:  # We do not prune BN term
#                 step += 1
#                 continue
#             # if(epoch > 0):
#             #     tensor = (param.data * spiking_list[cnt]).cpu().numpy()
#             #     cnt += 1
#             new_mask = np.where(abs(tensor) < percentile_value[cnt], 0, torch.FloatTensor([1]))
#             # new_mask = np.where(_normalize_scores(torch.from_numpy(tensor)**2).numpy() < percentile_value, 0, torch.FloatTensor([1]))
#             mask[step] = new_mask
#             step += 1
#             cnt += 1
#     step = 0

#     return  mask


# def simple_norm(x):
#     return (x - torch.min(x)) / (torch.max(x) - torch.min(x)) 

# def get_layerscore(model, lmask, mask):

#     if(lmask == None):
#         return([0 for i in range(len(mask))])

#     scores = []
#     step = 0
#     last_layer_weight = 0
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             if (len(param.data.shape)) == 1:  
#                 scores.append(0)
#                 step += 1
#                 continue
#             tensor = param.data.clone()
#             tensor = tensor*torch.from_numpy(lmask[step]).cuda()
#             if(len(tensor.shape) == 4):
#                 tensor = abs(tensor).sum(axis=[0,2,3])
#             elif(len(tensor.shape) == 2):
#                 tensor = abs(tensor).sum(axis=0)
#             else:
#                 print("Error Shape !")
#             scores.append(tensor)
#             step += 1
#             if(step == len(mask)):
#                 last_layer_weight = param.data.clone()
#     pos = [[2, 6], 0, 4, 0, 8, 0, 8, 0, 10, 0, 12, 0, 14, 0, [16, 20], 0, 18, 0, 22, 
#     0, 22, 0, 24, 0, 26, 0, 28, 0, [30, 34], 0, 32, 0, 36, 0, 36, 0, 38, 0, 40, 0, 41, -1]

#     layerscore = []
#     for i, x in enumerate(pos):
#         shape = [1 for k in range(len(mask[i].shape))]
#         shape[0] = -1
#         if(isinstance(x, list)):
#             layerscore.append(_normalize_scores((scores[x[0]] + scores[x[1]])**2).view(shape))
#             # layerscore.append(simple_norm(scores[x[0]] + scores[x[1]]).view(shape))

#         elif(x == 0):
#             layerscore.append(0)
#         elif(x == -1):
#             layerscore.append(0)
#             # layerscore.append(_normalize_scores(scores[-1]**2))
#             # layerscore.append(simple_norm(abs(last_layer_weight)))
#         else:
#             layerscore.append(_normalize_scores(scores[x]**2).view(shape))
#             # layerscore.append(simple_norm(scores[x]**2).view(shape))
#     return layerscore


# def get_pruning_maks_nextlayer(args, percent, mask, model,epoch, lmask=None):
#     step = 0
#     cnt = 0
#     all_param = []
#     # all_param_p = []
#     # all_param_n = []

#     layerscore = get_layerscore(model, lmask, mask)

#     step = 0
#     for (name, param) in model.named_parameters():
#         # print(name)
#         # We do not prune bias term
#         if 'weight' in name:
#             if (len(param.data.shape)) == 1:  # We do not prune BN term
#                 # print("SKIP:  " + name)
#                 step += 1
#                 continue
#             tensor = (abs(param.data)+layerscore[step]).cpu().numpy() #abs??
#             # tensor = simple_norm(simple_norm(abs(param.data))+layerscore[step]).cpu().numpy()
#             # if(epoch > 0):
#             #     tensor = (param.data * spiking_list[cnt]).cpu().numpy()
#             #     cnt += 1
#             alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
#             all_param.append(list(abs(alive)))
#             step += 1
            
#             # alive = _normalize_scores(torch.from_numpy(alive)**2).numpy() 
#             # all_param.append(list(alive)) 
#     # scores = []
#     # for w in all_param:
#     #   percentile_value = np.sort(w)[int(float(len(w)) / float(100. / percent))]
#     #   scores.append(scores)
#     param_whole = np.concatenate(all_param)
#     percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0]) / float(100. / percent))]

#             # pos = tensor > 0
#             # alive_p = tensor * pos
#             # alive_p = alive_p[np.nonzero(alive_p)]
#             # all_param_p.append(list(abs(alive_p)))
#             # neg = tensor < 0
#             # alive_n = tensor * neg
#             # alive_n = alive_n[np.nonzero(alive_n)]
#             # all_param_n.append(list(abs(alive_n)))

#     # param_whole_p = np.concatenate(all_param_p)
#     # percentile_value_p = np.sort(param_whole_p)[int(float(param_whole_p.shape[0]) / float(100. / percent))]
            
#     # param_whole_n = np.concatenate(all_param_n)
#     # percentile_value_n = np.sort(param_whole_n)[int(float(param_whole_n.shape[0]) / float(100. / percent))]



#     step = 0
#     cnt = 0

#     for name, param in model.named_parameters():
#         # We do not prune bias term
#         if 'weight' in name:
#             # tensor = param.data.cpu().numpy()
#             tensor = param.data
#             if (len(tensor.shape)) == 1:  # We do not prune BN term
#                 step += 1
#                 continue
#             # if(epoch > 0):
#             #     tensor = (param.data * spiking_list[cnt]).cpu().numpy()
#             #     cnt += 1
#             new_mask = np.where((abs(tensor) + layerscore[step]).cpu().numpy() < percentile_value, 0, torch.FloatTensor([1]))
#             # new_mask = np.where(simple_norm((simple_norm(abs(tensor))+layerscore[step])).cpu().numpy() < percentile_value, 0, torch.FloatTensor([1]))
#             # new_mask = np.where(_normalize_scores(torch.from_numpy(tensor)**2).numpy() < percentile_value, 0, torch.FloatTensor([1]))
#             # new_mask_p = np.where(abs(tensor*(tensor>0)) < percentile_value_p, 0, torch.FloatTensor([1]))
#             # new_mask_n = np.where(abs(tensor*(tensor<0)) < percentile_value_n, 0, torch.FloatTensor([1]))
#             # new_mask = new_mask_p + new_mask_n
#             mask[step] = new_mask
#             step += 1
#     step = 0

#     return  mask

# def get_pruning_maks(args, percent, mask, model,epoch, spiking_list=None):
#     step = 0
#     cnt = 0
#     all_param = []
#     # all_param_p = []
#     # all_param_n = []
#     cnt = 0
#     w_scal = 2.0
#     w_scal2 = 1.0
#     for name, param in model.named_parameters():
#         # print(name)
#         # We do not prune bias term
#         if 'weight' in name:
#             if (len(param.data.shape)) == 1:  # We do not prune BN term
#                 # print("SKIP:  " + name)
#                 cnt += 1
#                 continue
#             # if(len(param.data.shape)) == 4:
#             #     w_scal = w_scal * (1.0 - (0.0005 * cnt))  
#             #     tep_scal = w_scal
#             # else:
#             #     tep_scal = 0.8 * w_scal2
#             #     w_scal2 = w_scal2 * 1.5
#             weight = param.data
#             if len(param.data.shape) == 4:
#                 temp_weight = torch.abs(weight).sum(axis=[2,3], keepdim=True)
#                 temp_weight1 = temp_weight.sum(axis = [1], keepdim=True)
#                 temp_weight2 = temp_weight1.sum(axis = [0], keepdim=True)
#                 temp_weight =  torch.abs(weight) + (temp_weight/temp_weight1 + temp_weight1/temp_weight2)
#             elif len(param.data.shape) == 2:
#                 temp_weight1 = torch.abs(weight).sum(axis=1, keepdim=True)
#                 temp_weight2 = temp_weight1.sum()
#                 temp_weight = torch.abs(weight) + (temp_weight1/temp_weight2)
#             tensor = temp_weight.cpu().numpy()
#             # tensor = (param.data * tep_scal).cpu().numpy()
#             # if(epoch > 0):
#             #     tensor = (param.data * spiking_list[cnt]).cpu().numpy()
#             #     cnt += 1
#             alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
#             all_param.append(list(abs(alive)))
#             cnt += 1
            
#             # alive = _normalize_scores(torch.from_numpy(alive)**2).numpy() 
#             # all_param.append(list(alive)) 
#     # scores = []
#     # for w in all_param:
#     # 	percentile_value = np.sort(w)[int(float(len(w)) / float(100. / percent))]
#     # 	scores.append(scores)
#     param_whole = np.concatenate(all_param)
#     percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0]) / float(100. / percent))]

#             # pos = tensor > 0
#             # alive_p = tensor * pos
#             # alive_p = alive_p[np.nonzero(alive_p)]
#             # all_param_p.append(list(abs(alive_p)))
#             # neg = tensor < 0
#             # alive_n = tensor * neg
#             # alive_n = alive_n[np.nonzero(alive_n)]
#             # all_param_n.append(list(abs(alive_n)))

#     # param_whole_p = np.concatenate(all_param_p)
#     # percentile_value_p = np.sort(param_whole_p)[int(float(param_whole_p.shape[0]) / float(100. / percent))]
            
#     # param_whole_n = np.concatenate(all_param_n)
#     # percentile_value_n = np.sort(param_whole_n)[int(float(param_whole_n.shape[0]) / float(100. / percent))]



#     step = 0
#     cnt = 0
#     w_scal = 2.0
#     w_scal2 = 1.0

#     for name, param in model.named_parameters():
#         # We do not prune bias term
#         if 'weight' in name:
#             if (len(param.data.shape)) == 1:  # We do not prune BN term
#                 step += 1
#                 cnt += 1
#                 continue
#             # if(len(param.data.shape)) == 4:
#             #     w_scal = w_scal * (1.0 - (0.0005 * cnt))
#             #     tep_scal = w_scal
#             # else:
                
#             #     tep_scal = 0.8 * w_scal2
#             #     w_scal2 = w_scal2 * 1.5

#             weight = param.data
#             if len(param.data.shape) == 4:
#                 temp_weight = torch.abs(weight).sum(axis=[2,3], keepdim=True)
#                 temp_weight1 = temp_weight.sum(axis = [1], keepdim=True)
#                 temp_weight2 = temp_weight1.sum(axis = [0], keepdim=True)
#                 temp_weight =  torch.abs(weight) + (temp_weight/temp_weight1 + temp_weight1/temp_weight2)
#             elif len(param.data.shape) == 2:
#                 temp_weight1 = torch.abs(weight).sum(axis=1, keepdim=True)
#                 temp_weight2 = temp_weight1.sum()
#                 temp_weight = torch.abs(weight) + (temp_weight1/temp_weight2)
#             tensor = temp_weight.cpu().numpy()

#             tensor = param.data.cpu().numpy()
#             # tensor = (param.data * tep_scal).cpu().numpy()
#             # if(epoch > 0):
#             #     tensor = (param.data * spiking_list[cnt]).cpu().numpy()
#             #     cnt += 1
#             new_mask = np.where(abs(tensor) < percentile_value, 0, torch.FloatTensor([1]))
#             # new_mask = np.where(abs(tensor) < scores[step], 0, torch.FloatTensor([1]))
#             # new_mask = np.where(_normalize_scores(torch.from_numpy(tensor)**2).numpy() < percentile_value, 0, torch.FloatTensor([1]))
#             # new_mask_p = np.where(abs(tensor*(tensor>0)) < percentile_value_p, 0, torch.FloatTensor([1]))
#             # new_mask_n = np.where(abs(tensor*(tensor<0)) < percentile_value_n, 0, torch.FloatTensor([1]))
#             # new_mask = new_mask_p + new_mask_n
#             mask[step] = new_mask
#             step += 1
#             cnt += 1
#     step = 0

#     return  mask

# def get_pruning_maks_s(args, percent, mask, model,epoch, spiking_list=None):
#     step = 0
#     cnt = 0
#     all_param = []
#     for name, param in model.named_parameters():
#         # print(name)
#         # We do not prune bias term
#         if 'weight' in name:
#             if (len(param.data.shape)) == 1:  # We do not prune BN term
#                 # print("SKIP:  " + name)
#                 continue
#             tensor = param.data.cpu().numpy()
#             if(epoch > 0):
#                 tensor = (param.data * _normalize_scores(spiking_list[cnt])).cpu().numpy()
#                 print(_normalize_scores(spiking_list[cnt]).mean())
#                 cnt += 1
#             alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
#             if epoch == 0:
#                 all_param.append(list(abs(alive)))
#             else:
#                 # alive = _normalize_scores(torch.from_numpy(alive)**2).numpy() 
#                 all_param.append(list(abs(alive)))

#     param_whole = np.concatenate(all_param)
#     percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0]) / float(100. / percent))]

#     step = 0
#     cnt = 0

#     for name, param in model.named_parameters():
#         # We do not prune bias term
#         if 'weight' in name:
#             tensor = param.data.cpu().numpy()
#             if (len(tensor.shape)) == 1:  # We do not prune BN term
#                 step += 1
#                 continue
#             if(epoch > 0):
#                 tensor = (param.data * _normalize_scores(spiking_list[cnt])).cpu().numpy()
#                 cnt += 1
#             if (epoch == 0):
#                 new_mask = np.where(abs(tensor) < percentile_value, 0, torch.FloatTensor([1]))
#             else:
#                 # new_mask = np.where(_normalize_scores(torch.from_numpy(tensor)**2).numpy() < percentile_value, 0, torch.FloatTensor([1]))
#                 new_mask = np.where(abs(tensor) < percentile_value, 0, torch.FloatTensor([1]))
#             mask[step] = new_mask
#             step += 1
#     step = 0

#     return  mask

# def original_initialization(mask_temp, initial_state_dict, model):

#     step = 0
#     for name, param in model.named_parameters():
#         if "weight" in name:
#             weight_dev = param.device
#             param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
#             # param.data = torch.from_numpy(mask_temp[step] * param.data.cpu().numpy()).to(weight_dev)
#             # print(param.data.sum() / mask_temp[step].sum())
#             # init(mask_temp[step], param.data, name)
#             # print(param.data.sum() / mask_temp[step].sum())

#             step = step + 1
#         if "bias" in name:
#             # pass
#             param.data = initial_state_dict[name].to(param.device)
#     step = 0

#     return model

def original_initialization_nobias(mask_temp, initial_state_dict, model):

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name] +1

    step = 0

    return model


# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            # output = sum(output)
            loss = F.cross_entropy(output, target).item()  # sum up batch loss
            test_loss += loss
            test_losses.append(loss)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        # print("test_loss: ", test_loss)
    return accuracy, test_loss


def test_ann(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            # reset_net(model)

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy




def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)

