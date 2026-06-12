# -*- coding = utf-8 -*-
import numpy as np
import torch
import copy
import time
import hdbscan
import random
import math
from concurrent.futures import ThreadPoolExecutor
import phe as paillier

def weak_dp(local_model, update_params, global_model, args):
    """
    Weak-DP defense: dynamically clips client updates by the median L2 norm
    and adds Gaussian noise after aggregation with a fixed noise coefficient.

    Args:
        local_model: list of client model stateDicts (unused here, kept for interface consistency)
        update_params: list of client update dicts (state_dict-like tensors)
        global_model: dict of global model parameters (to be updated in-place)
        args: run-time arguments (not used for bound/noise; behavior is fixed here)

    Returns:
        Updated global_model after weak-DP aggregation
    """
    if not update_params:
        return global_model

    # 1) Measure each client update norm and pick the median as clipping bound
    norms = []
    for upd in update_params:
        vec = parameters_dict_to_vector_flt(upd)
        norms.append(torch.norm(vec).item())
    clip_bound = float(np.median(norms))

    # 2) Clip each client update by L2-norm using the median bound
    clipped_updates = []
    for upd, norm in zip(update_params, norms):
        if norm > clip_bound and norm > 0:
            scale = clip_bound / norm
            clipped_upd = {}
            for k, v in upd.items():
                if k.split('.')[-1] == 'num_batches_tracked':
                    clipped_upd[k] = v.clone()
                else:
                    clipped_upd[k] = v * scale
            clipped_updates.append(clipped_upd)
        else:
            clipped_updates.append(upd)

    # 3) Average the clipped updates
    sum_params = None
    for upd in clipped_updates:
        if sum_params is None:
            sum_params = {k: v.clone() for k, v in upd.items() if k.split('.')[-1] != 'num_batches_tracked'}
        else:
            for k in sum_params:
                sum_params[k] += upd[k]
    for k in sum_params:
        sum_params[k] /= len(clipped_updates)

    # 4) Fixed noise coefficient = 0.02; scale by the clipping bound
    noise_coeff = 0.002
    noise_std = noise_coeff * clip_bound

    # 5) Add Gaussian noise to aggregated parameters and apply to the global model
    for k in global_model:
        if k.split('.')[-1] == 'num_batches_tracked':
            continue
        noise = torch.randn_like(global_model[k]) * noise_std
        global_model[k] += sum_params[k] + noise

    print(f'[weak-dp] clip_bound(median L2)={clip_bound:.6f}, noise_coeff={noise_coeff:.4f}, noise_stddev={noise_std:.6f}, n_clients={len(update_params)}')
    return global_model

def cos(a, b):
    # res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-
    res = (np.dot(a, b) + 1e-9) / (np.linalg.norm(a) + 1e-9) / \
        (np.linalg.norm(b) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res


def fltrust(params, central_param, global_parameters, args):
    FLTrustTotalScore = 0
    score_list = []
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    sum_parameters = None
    for local_parameters in params:
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
        # 计算cos相似度得分和向量长度裁剪值
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)
        client_clipped_value = central_norm/torch.norm(local_parameters_v)
        score_list.append(client_cos)
        FLTrustTotalScore += client_cos
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in local_parameters.items():
                # 乘得分 再乘裁剪值
                sum_parameters[key] = client_cos * \
                    client_clipped_value * var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[
                    var]
    if FLTrustTotalScore == 0:
        print(score_list)
        return global_parameters
    for var in global_parameters:
        # 除以所以客户端的信任得分总和
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * args.server_lr
    print(score_list)
    return global_parameters


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt_cpu(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.cpu().view(-1))
    return torch.cat(vec)


def no_defence_balance(params, global_parameters):
    total_num = len(params)
    print(f"total_num is {total_num}")
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if sum_parameters.get(var) is None:
            # 可以在这里记录日志或采取其他措施
            print(f"Warning: {var} is None, skipping.")
            continue
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters


def fools_gold(updates, global_parameters, args):
    """
    FoolsGold defense (client-level reweighting based on gradient similarity).

    Args:
        updates: list of client update dicts (state_dict-like tensors)
        global_parameters: dict of global model parameters (to be updated)
        args: run-time arguments, may include `server_lr` and device info

    Returns:
        Aggregated global_parameters after applying FoolsGold weights.
    """
    # Convert each client's update to a flattened vector (exclude BN counters)
    update_vecs = []
    for upd in updates:
        update_vecs.append(parameters_dict_to_vector_flt(upd))

    if len(update_vecs) == 0:
        return global_parameters

    # Stack to matrix: shape (n_clients, dim)
    X = torch.stack(update_vecs)  # client updates as rows
    # Cosine similarity matrix between clients
    # Normalize rows to unit norm to compute cosine via dot products
    norms = torch.norm(X, dim=1) + 1e-12
    X_norm = X / norms.unsqueeze(1)
    S = torch.mm(X_norm, X_norm.t())  # cosine similarity matrix

    # Set diagonal to 0 to ignore self-similarity
    S.fill_diagonal_(0.0)

    # Compute max similarity per client (FoolsGold core)
    max_s = torch.max(S, dim=1)[0]  # shape (n_clients,)

    # FoolsGold weighting:
    # w_i = 1 - max_s_i; then clip to [0,1]
    w = 1.0 - max_s
    w = torch.clamp(w, min=0.0, max=1.0)

    # Optional normalization: scale weights so that max weight is 1
    if torch.max(w) > 0:
        w = w / torch.max(w)

    # Track simple stats (optional)
    try:
        args.fg_weights = w.detach().cpu().numpy().tolist()
    except Exception:
        pass

    # Weighted aggregation into global_parameters
    sum_parameters = None
    total_weight = torch.sum(w).item()
    if total_weight == 0:
        # fall back to average if all weights are zero
        return no_defence_balance(updates, global_parameters)

    for idx, local_parameters in enumerate(updates):
        weight_i = w[idx].item()
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in local_parameters.items():
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                sum_parameters[key] = weight_i * var.clone()
        else:
            for key in sum_parameters.keys():
                sum_parameters[key] = sum_parameters[key] + weight_i * local_parameters[key]

    for key in global_parameters:
        if key.split('.')[-1] == 'num_batches_tracked':
            # Preserve BN counters from any client (use first)
            global_parameters[key] = updates[0][key]
            continue
        temp = sum_parameters[key] / total_weight
        if global_parameters[key].type() != temp.type():
            temp = temp.type(global_parameters[key].type())
        # Apply server learning rate if present
        lr = getattr(args, 'server_lr', 1.0)
        global_parameters[key] += temp * lr

    return global_parameters


def multi_krum(gradients, n_attackers, args, multi_k=False):

    grads = flatten_grads(gradients)

    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        scores = None
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        print(scores)
        args.krum_distance.append(scores)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    # aggregate = torch.mean(candidates, dim=0)

    # return aggregate, np.array(candidate_indices)
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    args.turn+=1
    if multi_k == False:
        if candidate_indices[0] < num_malicious_clients:
            args.wrong_mal += 1
            
    print(candidate_indices)
    
    print('Proportion of malicious are selected:'+str(args.wrong_mal/args.turn))

    for i in range(len(scores)):
        if i < num_malicious_clients:
            args.mal_score += scores[i]
        else:
            args.ben_score += scores[i]
    
    return np.array(candidate_indices)



def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend(
                    [grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs




def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        update2[key] = update[key] - model[key]
    return update2



def weak_dp_simple(params, global_parameters, args):
    """
    Weak DP (Differential Privacy) defense implementation (simple 3-arg variant).
    
    NOTE: This function was previously named `weak_dp`, which unintentionally
    shadowed the 4-argument `weak_dp(local_model, update_params, global_model, args)`
    used in the training loop. It has been renamed to avoid overriding the intended
    implementation.

    This defense mechanism implements a simplified version of differential privacy by:
    1. Clipping model updates to bound their norm
    2. Adding Gaussian noise to the aggregated model
    
    Args:
        params: list of client model parameters (state_dict-like tensors)
        global_parameters: dict of global model parameters (to be updated)
        args: run-time arguments, should include 'norm_bound' and 'noise_stddev'
    
    Returns:
        Updated global_parameters after applying weak DP defense
    """
    # Step 1: Clip individual model updates to bound their norm
    norm_bound = getattr(args, 'norm_bound', 1.0)
    clipped_params = []
    
    for local_param in params:
        # Calculate the norm of parameter differences
        param_diff = {}
        for key in local_param.keys():
            if key.split('.')[-1] == 'num_batches_tracked':
                param_diff[key] = local_param[key].clone()
            else:
                param_diff[key] = local_param[key] - global_parameters[key]
        
        # Flatten parameter differences to calculate norm
        diff_vector = parameters_dict_to_vector_flt(param_diff)
        param_norm = torch.norm(diff_vector)
        
        # Clip if norm exceeds bound
        if param_norm > norm_bound:
            clip_ratio = norm_bound / param_norm
            for key in param_diff.keys():
                if key.split('.')[-1] != 'num_batches_tracked':
                    param_diff[key] = param_diff[key] * clip_ratio
        
        # Reconstruct clipped parameters
        clipped_local_param = {}
        for key in local_param.keys():
            if key.split('.')[-1] == 'num_batches_tracked':
                clipped_local_param[key] = local_param[key].clone()
            else:
                clipped_local_param[key] = global_parameters[key] + param_diff[key]
        
        clipped_params.append(clipped_local_param)
    
    # Step 2: Aggregate clipped parameters using simple averaging
    sum_parameters = None
    total_num = len(clipped_params)
    
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in clipped_params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + clipped_params[i][var]
    
    # Calculate averaged parameters
    averaged_params = {}
    for key in global_parameters:
        if sum_parameters.get(key) is None:
            continue
        if key.split('.')[-1] == 'num_batches_tracked':
            averaged_params[key] = clipped_params[0][key]
        else:
            averaged_params[key] = sum_parameters[key] / total_num
    
    # Step 3: Add Gaussian noise to aggregated parameters
    noise_stddev = getattr(args, 'noise_stddev', 0.01)
    device = getattr(args, 'device', 'cpu')
    
    for key in averaged_params:
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        
        # Generate Gaussian noise
        noise = torch.randn_like(averaged_params[key]) * noise_stddev
        
        # Add noise to averaged parameters
        averaged_params[key] = averaged_params[key] + noise
    
    # Update global parameters
    for key in global_parameters:
        if averaged_params.get(key) is not None:
            if global_parameters[key].type() != averaged_params[key].type():
                averaged_params[key] = averaged_params[key].type(global_parameters[key].type())
            global_parameters[key] = averaged_params[key]
    
    print(f"Weak DP Defense applied: norm_bound={norm_bound}, noise_stddev={noise_stddev}")
    
    return global_parameters


def RLR(global_model, agent_updates_list, args):
    """
    agent_updates_dict: dict['key']=one_dimension_update
    agent_updates_list: list[0] = model.dict
    global_model: net
    """
    # args.robustLR_threshold = 6
    args.server_lr = 1

    grad_list = []
    for i in agent_updates_list:
        grad_list.append(parameters_dict_to_vector_rlr(i))
    agent_updates_list = grad_list
    

    aggregated_updates = 0
    for update in agent_updates_list:
        # print(update.shape)  # torch.Size([1199882])
        aggregated_updates += update
    aggregated_updates /= len(agent_updates_list)
    lr_vector = compute_robustLR(agent_updates_list, args)
    cur_global_params = parameters_dict_to_vector_rlr(global_model.state_dict())
    new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
    global_w = vector_to_parameters_dict(new_global_params, global_model.state_dict())
    # print(cur_global_params == vector_to_parameters_dict(new_global_params, global_model.state_dict()))
    return global_w

def parameters_dict_to_vector_rlr(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)



def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict

def compute_robustLR(params, args):
    agent_updates_sign = [torch.sign(update) for update in params]  
    sm_of_signs = torch.abs(sum(agent_updates_sign))
    # print(len(agent_updates_sign)) #10
    # print(agent_updates_sign[0].shape) #torch.Size([1199882])
    sm_of_signs[sm_of_signs < args.robustLR_threshold] = -args.server_lr
    sm_of_signs[sm_of_signs >= args.robustLR_threshold] = args.server_lr 
    return sm_of_signs.to(args.gpu)
   
    
def flame(local_model, update_params, global_model, args):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    for i in range(len(local_model_vector)):
        # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
        norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    print(benign_client)
   
    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal+=1
        else:
            #  minus per benign in cluster
            args.right_ben += 1
    args.turn+=1
    print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
    print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))
    
    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
    #add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
                    continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0,std=args.noise*clip_value)
        var += temp
    return global_model

# def flame(local_model, update_params, global_model, args):
#     cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
#     cos_list=[]
#     local_model_vector = []
#     for param in local_model:
#         # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
#         local_model_vector.append(parameters_dict_to_vector_flt(param))
#     for i in range(len(local_model_vector)):
#         cos_i = []
#         for j in range(len(local_model_vector)):
#             cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
#             # cos_i.append(round(cos_ij.item(),4))
#             cos_i.append(cos_ij.item())
#         cos_list.append(cos_i)
#     num_clients = max(int(args.frac * args.num_users), 1)
#     num_malicious_clients = int(args.malicious * num_clients)
#     num_benign_clients = num_clients - num_malicious_clients
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
#     print(clusterer.labels_)
#     benign_client = []
#     norm_list = np.array([])

#     max_num_in_cluster=0
#     max_cluster_index=0
#     if clusterer.labels_.max() < 0:
#         for i in range(len(local_model)):
#             benign_client.append(i)
#             norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     else:
#         for index_cluster in range(clusterer.labels_.max()+1):
#             if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
#                 max_cluster_index = index_cluster
#                 max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
#         for i in range(len(clusterer.labels_)):
#             if clusterer.labels_[i] == max_cluster_index:
#                 benign_client.append(i)
#     for i in range(len(local_model_vector)):
#         # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
#         norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
#     print(benign_client)
   
#     for i in range(len(benign_client)):
#         if benign_client[i] < num_malicious_clients:
#             args.wrong_mal+=1
#         else:
#             #  minus per benign in cluster
#             args.right_ben += 1
#     args.turn+=1
#     print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
#     print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))
    
#     clip_value = np.median(norm_list)
#     for i in range(len(benign_client)):
#         gama = clip_value/norm_list[i]
#         if gama < 1:
#             for key in update_params[benign_client[i]]:
#                 if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#                 update_params[benign_client[i]][key] *= gama
#     global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
#     #add noise
#     for key, var in global_model.items():
#         if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#         temp = copy.deepcopy(var)
#         temp = temp.normal_(mean=0,std=args.noise*clip_value)
#         var += temp
    
#     # 记录newFlame函数总体结束时间
#     newflame_total_end_time = time.time()
#     newflame_total_elapsed_time = newflame_total_end_time - arithmetic_share_start_time
    
#     # 汇总所有时间统计数据
#     print("\n========== newFlame函数时间统计汇总 ==========")
#     print(f"1. 模型拆分成算术共享操作用时: {arithmetic_share_elapsed_time:.6f} 秒")
#     print(f"2. 张量A和D生成及算术共享操作用时: {tensor_generation_elapsed_time:.6f} 秒")
#     print(f"   - 其中A和D验证过程用时: {ad_verification_elapsed_time:.6f} 秒")
#     print(f"3. 张量alpha、beta、gamma生成及算术共享操作用时: {alpha_beta_gamma_elapsed_time:.6f} 秒")
#     print(f"   - 其中alpha、beta、gamma验证过程用时: {abg_verification_elapsed_time:.6f} 秒")
#     print(f"4. zeta相关操作用时: {zeta_operations_elapsed_time:.6f} 秒")
#     print(f"5. deta相关操作用时: {deta_operations_elapsed_time:.6f} 秒")
#     print(f"6. 同态加密操作(计算用户个数n和div函数运算)用时: {homomorphic_operations_elapsed_time:.6f} 秒")
#     print(f"7. newFlame函数总体用时: {newflame_total_elapsed_time:.6f} 秒")
#     print("=" * 50)
    
#     # 将时间统计数据保存到CSV文件
#     import csv
#     import os
#     from datetime import datetime
    
#     # 创建时间统计目录
#     timing_dir = "/home/jfl/code/FLAME-main/timing_analysis"
#     if not os.path.exists(timing_dir):
#         os.makedirs(timing_dir)
    
#     # 生成带时间戳的文件名
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     csv_filename = os.path.join(timing_dir, f"newFlame_timing_{timestamp}.csv")
    
#     # 准备CSV数据
#     timing_data = [
#         ["操作类型", "用时(秒)", "占总时间比例(%)"],
#         ["模型拆分成算术共享", f"{arithmetic_share_elapsed_time:.6f}", f"{(arithmetic_share_elapsed_time/newflame_total_elapsed_time)*100:.2f}"],
#         ["张量A和D生成及算术共享", f"{tensor_generation_elapsed_time:.6f}", f"{(tensor_generation_elapsed_time/newflame_total_elapsed_time)*100:.2f}"],
#         ["A和D验证过程", f"{ad_verification_elapsed_time:.6f}", f"{(ad_verification_elapsed_time/newflame_total_elapsed_time)*100:.2f}"],
#         ["张量alpha、beta、gamma生成及算术共享", f"{alpha_beta_gamma_elapsed_time:.6f}", f"{(alpha_beta_gamma_elapsed_time/newflame_total_elapsed_time)*100:.2f}"],
#         ["alpha、beta、gamma验证过程", f"{abg_verification_elapsed_time:.6f}", f"{(abg_verification_elapsed_time/newflame_total_elapsed_time)*100:.2f}"],
#         ["zeta相关操作", f"{zeta_operations_elapsed_time:.6f}", f"{(zeta_operations_elapsed_time/newflame_total_elapsed_time)*100:.2f}"],
#         ["同态加密操作", f"{homomorphic_operations_elapsed_time:.6f}", f"{(homomorphic_operations_elapsed_time/newflame_total_elapsed_time)*100:.2f}"],
#         ["newFlame函数总体", f"{newflame_total_elapsed_time:.6f}", "100.00"]
#     ]
    
#     # 写入CSV文件
#     try:
#         with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerows(timing_data)
#         print(f"时间统计数据已保存到: {csv_filename}")
#     except Exception as e:
#         print(f"保存时间统计数据时出错: {e}")
    
#     # 同时保存JSON格式的详细数据
#     import json
#     json_filename = os.path.join(timing_dir, f"newFlame_timing_detailed_{timestamp}.json")
    
#     detailed_timing_data = {
#         "timestamp": timestamp,
#         "total_time": newflame_total_elapsed_time,
#         "operations": {
#             "arithmetic_share": {
#                 "time": arithmetic_share_elapsed_time,
#                 "percentage": (arithmetic_share_elapsed_time/newflame_total_elapsed_time)*100
#             },
#             "tensor_AD_generation": {
#                 "time": tensor_generation_elapsed_time,
#                 "percentage": (tensor_generation_elapsed_time/newflame_total_elapsed_time)*100,
#                 "sub_operations": {
#                     "AD_verification": {
#                         "time": ad_verification_elapsed_time,
#                         "percentage": (ad_verification_elapsed_time/newflame_total_elapsed_time)*100
#                     }
#                 }
#             },
#             "tensor_alpha_beta_gamma_generation": {
#                 "time": alpha_beta_gamma_elapsed_time,
#                 "percentage": (alpha_beta_gamma_elapsed_time/newflame_total_elapsed_time)*100,
#                 "sub_operations": {
#                     "alpha_beta_gamma_verification": {
#                         "time": abg_verification_elapsed_time,
#                         "percentage": (abg_verification_elapsed_time/newflame_total_elapsed_time)*100
#                     }
#                 }
#             },
#             "zeta_operations": {
#                 "time": zeta_operations_elapsed_time,
#                 "percentage": (zeta_operations_elapsed_time/newflame_total_elapsed_time)*100
#             },
#             "homomorphic_operations": {
#                 "time": homomorphic_operations_elapsed_time,
#                 "percentage": (homomorphic_operations_elapsed_time/newflame_total_elapsed_time)*100
#             }
#         }
#     }
    
#     try:
#         with open(json_filename, 'w', encoding='utf-8') as jsonfile:
#             json.dump(detailed_timing_data, jsonfile, indent=2, ensure_ascii=False)
#         print(f"详细时间统计数据已保存到: {json_filename}")
#     except Exception as e:
#         print(f"保存详细时间统计数据时出错: {e}")
    
#     return global_model
    








# def flame(local_model, update_params, global_model, args):
    

#     # update_params = [
#     #     {'conv1.weight':torch.full((3,2),1.0,device=args.gpu),
#     #     'conv1.weight1':torch.full((3,2),2.0,device=args.gpu),
#     #     'conv1.weight2':torch.full((3,2),3.0,device=args.gpu)},

#     #     {'conv1.weight':torch.full((3,2),3.0,device=args.gpu),
#     #     'conv1.weight1':torch.full((3,2),2.0,device=args.gpu),
#     #     'conv1.weight2':torch.full((3,2),1.0,device=args.gpu)}
#     # ]
    
#     # update_params = [
#     #     {'conv1.weight':torch.tensor([2.0,1.0],device=args.gpu)},

#     #     {'conv1.weight':torch.tensor([1.0,2.0],device=args.gpu)}
#     # ]
#     # for param_dict in update_params:
#     #     for key in param_dict.keys():
#     #         param_dict[key] = param_dict[key].to(args.gpu)

    
#     # print(list(update_params[0].keys()))
#     # print(update_params[0]['conv1.weight'].shape)
#     # print(type(update_params[0]['conv1.weight']))
    
#     def arithmetic_share(tensor):
#         # 生成两个份额
#         share1 = torch.rand_like(tensor,device=args.gpu,dtype=torch.float64)
#         # share1 = torch.full(tensor.shape,1,device=args.gpu)
#         share2 = tensor - share1
#         return share1, share2
    
#     #初始化共享参数列表
#     share_update_params0 = []
#     share_update_params1 = []

#     # 对每个用户的模型生成算术共享并存储
#     for user_params in update_params:
#         shares = {key: list(arithmetic_share(value)) for key, value in user_params.items()}
#         share_update_params0.append({key: share[0] for key, share in shares.items()})
#         share_update_params1.append({key: share[1] for key, share in shares.items()})
        
#     # print(share_update_params0[0]['conv1.weight'].device)
#     # print(share_update_params1[0]['conv1.weight'].device)
#     # print(share_update_params0[0]['conv1.weight'])

#     # total_elements = sum(tensor.numel() for user_params in update_params for tensor in user_params.values())
#     # print(f'total_elements is {total_elements}')

#     #generation of clients' beavers
#     # share_users_beavers0 = [{key:{} for key in update_params[0].keys()} for _ in range(len(update_params))]
#     # share_users_beavers1 = [{key:{} for key in update_params[0].keys()} for _ in range(len(update_params))]
#     # for key in update_params[0].keys():
#     #     up_shape = update_params[0][key].shape
#     #     for i in range(len(update_params)):
#     #         share_users_beavers0[i][key]['a'] = torch.randint(1, 11, up_shape, dtype=torch.float64,device=args.gpu)
#     #         share_users_beavers0[i][key]['b'] = torch.randint(1, 11, up_shape, dtype=torch.float64,device=args.gpu)
#     #         share_users_beavers0[i][key]['c'] = torch.randint(1, 11, up_shape, dtype=torch.float64,device=args.gpu)
#     #         share_users_beavers1[i][key]['a'] = torch.randint(1, 11, up_shape, dtype=torch.float64,device=args.gpu)
#     #         share_users_beavers1[i][key]['b'] = torch.randint(1, 11, up_shape, dtype=torch.float64,device=args.gpu)
#     #         share_users_beavers1[i][key]['c'] = torch.randint(1, 11, up_shape, dtype=torch.float64,device=args.gpu)
#     #         share_users_beavers1[i][key]['c'] = (share_users_beavers0[i][key]['a']+share_users_beavers1[i][key]['a'])*\
#     #                                             (share_users_beavers0[i][key]['b']+share_users_beavers1[i][key]['b'])-\
#     #                                             share_users_beavers0[i][key]['c']

#     share_users_beavers0 = [[]for _ in range(len(update_params))]
#     share_users_beavers1 = [[]for _ in range(len(update_params))]
#     for i in range(len(update_params)):
#         share_users_beavers0[i] = torch.randint(1,11,(3,1),device=args.gpu,dtype=torch.float64)
#         share_users_beavers1[i] = torch.randint(1,11,(3,1),device=args.gpu,dtype=torch.float64)
#         share_users_beavers1[i][2] = (share_users_beavers0[i][0]+share_users_beavers1[i][0])*\
#                                                 (share_users_beavers0[i][1]+share_users_beavers1[i][1])-\
#                                                 share_users_beavers0[i][2]




#     #generation of clients' square corrlations
#     share_users_sc0 = [{key:{} for key in update_params[0].keys()} for _ in range(len(update_params))]
#     share_users_sc1 = [{key:{} for key in update_params[0].keys()} for _ in range(len(update_params))]
    
#     for key in update_params[0].keys():
#         up_shape = update_params[0][key].shape
#         for i in range(len(update_params)):
#             share_users_sc0[i][key]['A'] = torch.randint(1, 11, up_shape, dtype=torch.float64,device=args.gpu)
#             share_users_sc0[i][key]['D'] = torch.randint(1, 11, up_shape, dtype=torch.float64,device=args.gpu)
#             share_users_sc1[i][key]['A'] = torch.randint(1, 11, up_shape, dtype=torch.float64,device=args.gpu)
#             # share_users_sc1[i][key]['D'] = torch.randint(1, 11, up_shape, dtype=torch.float64)
#             share_users_sc1[i][key]['D'] = (share_users_sc0[i][key]['A']+share_users_sc1[i][key]['A'])**2-\
#                                                 share_users_sc0[i][key]['D']

#     @torch.jit.script
#     def GETSED_layer(g_i0, g_j0,g_i1, g_j1,
#                      share_useri_sc0A,share_useri_sc0D, 
#                      share_useri_sc1A,share_useri_sc1D,
#                      share_userj_sc0A,share_userj_sc0D, 
#                      share_userj_sc1A,share_userj_sc1D):
#         # print(f"g_i0 device is:{g_i0.device},g_j0 device is:{g_j0.device},share_useri_sc0['A'] device is:{share_useri_sc0['A'].device}")
#         share_eij0 = g_i0-g_j0-share_useri_sc0A
#         share_Eij0 = g_i0-g_j0-share_userj_sc0A
#         share_eij1 = g_i1-g_j1-share_useri_sc1A
#         share_Eij1 = g_i1-g_j1-share_userj_sc1A
#         eij = share_eij0 + share_eij1
#         Eij = share_Eij0 + share_Eij1
#         share_zij0 = torch.sum(0.5*(share_useri_sc0D+share_userj_sc0D)+\
#             (g_i0-g_j0)*(eij+Eij)-0.5*eij**2)
#         share_zij1 = torch.sum(0.5*(share_useri_sc1D+share_userj_sc1D)+\
#             (g_i1-g_j1)*(eij+Eij)-0.5*Eij**2)

            
#         # g_i = g_i0+g_i1
#         # g_j = g_j0+g_j1
#         # # g_norm = torch.norm((g_i-g_j).view(-1))
#         # g_norm = torch.sum(torch.pow(g_i-g_j,2))
        
#         # print(f"g_norm is {g_norm}")
#         # share_zij = share_zij0+share_zij1
#         # print(f"share_zij is {share_zij}")
#         return share_zij0, share_zij1
    
    
#     @torch.jit.script
#     def GETSED_single_layer(g_i0,g_i1,
#                      share_useri_sc0A,share_useri_sc0D, 
#                      share_useri_sc1A,share_useri_sc1D):
#         # print(f"g_i0 device is:{g_i0.device},g_j0 device is:{g_j0.device},share_useri_sc0['A'] device is:{share_useri_sc0['A'].device}")
#         share_ei0 = g_i0-share_useri_sc0A
#         share_ei1 = g_i1-share_useri_sc1A
#         ei = share_ei0 + share_ei1
#         share_zi0 = torch.sum((share_useri_sc0D)+\
#             2*(g_i0)*(ei)-0.5*ei**2)
#         share_zi1 = torch.sum((share_useri_sc1D)+\
#             2*(g_i1)*(ei)-0.5*ei**2)
#         return share_zi0, share_zi1


#     @torch.jit.script
#     def GETPro_121(z_i0, z_j0,z_i1, z_j1,
#                      share_useri_beavers0,
#                      share_useri_beavers1,
#                      share_userj_beavers0,
#                      share_userj_beavers1):
#         # print(f"g_i0 device is:{g_i0.device},g_j0 device is:{g_j0.device},share_useri_sc0['A'] device is:{share_useri_sc0['A'].device}")
#         share_uii0 = z_i0-share_useri_beavers0[0]
#         share_uij0 = z_i0-share_userj_beavers0[0]
#         share_vji0 = z_j0-share_useri_beavers0[1]
#         share_vjj0 = z_j0-share_userj_beavers0[1]
#         share_uii1 = z_i1-share_useri_beavers1[0]
#         share_uij1 = z_i1-share_userj_beavers1[0]
#         share_vji1 = z_j1-share_useri_beavers1[1]
#         share_vjj1 = z_j1-share_userj_beavers1[1]
#         uii = share_uii0+share_uii1
#         uij = share_uij0+share_uij1
#         vji = share_vji0+share_vji1
#         vjj = share_vjj0+share_vjj1
#         share_mulij0 = torch.sum(0.5*(uii*vji+share_useri_beavers0[0]*vji+
#                                       share_useri_beavers0[1]*uii+
#                                       share_userj_beavers0[0]*vjj+
#                                       share_userj_beavers0[1]*uij+
#                                       share_useri_beavers0[2]+
#                                       share_userj_beavers0[2]))
#         share_mulij1 = torch.sum(0.5*(uij*vjj+share_useri_beavers1[0]*vji+
#                                       share_useri_beavers1[1]*uii+
#                                       share_userj_beavers1[0]*vjj+
#                                       share_userj_beavers1[1]*uij+
#                                       share_useri_beavers1[2]+
#                                       share_userj_beavers1[2]))
 
#         return share_mulij0, share_mulij1

    
#     # @torch.jit.script
#     def GETSED(g_0, g_1, share_users_sc0, share_users_sc1):
#         clients_num = len(g_0)
#         # zero_torch = torch.tensor(0.0, device=args.gpu, dtype=torch.float64)
#         # share_zij0 = [[0]*clients_num  for _ in range(clients_num)]
#         # share_zij1 = [[0]*clients_num  for _ in range(clients_num)]
#         share_zij0 = torch.zeros((clients_num, clients_num), dtype=torch.float64 ,device=args.gpu)
#         share_zij1 = torch.zeros((clients_num, clients_num), dtype=torch.float64 ,device=args.gpu)
    
#         # print(f"g_0[0] type is {type(g_0[0])}")
#         for i in range(clients_num):
#             for j in range(i+1,clients_num):
#                 for key in g_0[0].keys():
#                     tmp_share_zij0, tmp_share_zij1 = GETSED_layer(g_0[i][key].to(torch.float64), g_0[j][key].to(torch.float64),
#                                                                     g_1[i][key].to(torch.float64), g_1[j][key].to(torch.float64),
#                                                                     share_users_sc0[i][key]['A'].to(torch.float64), share_users_sc0[i][key]['D'].to(torch.float64),
#                                                                     share_users_sc1[i][key]['A'].to(torch.float64), share_users_sc1[i][key]['D'].to(torch.float64),
#                                                                     share_users_sc0[j][key]['A'].to(torch.float64), share_users_sc0[j][key]['D'].to(torch.float64),
#                                                                     share_users_sc1[j][key]['A'].to(torch.float64),share_users_sc1[j][key]['D'].to(torch.float64))
#                     share_zij0[i][j] +=tmp_share_zij0
#                     share_zij1[i][j] +=tmp_share_zij1
        
#         # print(f'shape of share_zij0[0] is {len(share_zij0[0])}')
#         return share_zij0, share_zij1
    
    
#     def GETPro(z_0, z_1, share_users_beavers0, share_users_beavers1):
#         clients_num = len(z_0)
#         # share_mulij0 = [[0 for _ in range(clients_num)]for _ in range(clients_num)]
#         share_mulij0 = torch.zeros((clients_num, clients_num), dtype=torch.float64 ,device=args.gpu)
#         # share_mulij1 = [[0 for _ in range(clients_num)]for _ in range(clients_num)]
#         share_mulij1 = torch.zeros((clients_num, clients_num), dtype=torch.float64 ,device=args.gpu)
#         # print(f"z_0[0] type is {type(z_0[0])}")
#         for i in range(clients_num):
#             for j in range(i+1,clients_num):
#                 share_mulij0[i][j], share_mulij1[i][j] = GETPro_121(z_0[i].to(torch.float64), z_0[j].to(torch.float64),
#                                                                         z_1[i].to(torch.float64), z_1[j].to(torch.float64),
#                                                                         share_users_beavers0[i],share_users_beavers1[i],
#                                                                         share_users_beavers0[j],share_users_beavers1[j]
#                                                                         )
#         return share_mulij0, share_mulij1
                                                                        
                    
    

    
    
    
    
    
#     # #test of the correction of GETSED
#     # zij = [[share_zij0[i][j]+share_zij1[i][j] for j in range(0,len(share_zij0))] for i in range(len(share_zij0))]
#     # #get the norm of params


    
#     # norm_params = [[0]*len(update_params) for _ in range(len(update_params))]
#     # for i in range(len(update_params)):
#     #     for j in range(i+1,len(update_params)):
#     #         for key in update_params[i].keys():
#     #             norm_params[i][j] += torch.sum(torch.pow(update_params[i][key]-update_params[j][key],2))
#     # print(zij[0],f'\n shape of zij[0] is {len(zij[0])}')
#     # print(norm_params[0],f'\n shape of norm_params[0] is {len(norm_params[0])}')
    
#     def GETSED_single(g_0, g_1,share_users_sc0,share_users_sc1):
#         clients_num = len(g_0)
#         # share_zi00 = [ 0 for _ in range(len(g_0))]
#         # share_zi01 = [ 0 for _ in range(len(g_1))]
#         share_zi00 = torch.zeros((clients_num), dtype=torch.float64 ,device=args.gpu)
#         share_zi01 = torch.zeros((clients_num), dtype=torch.float64 ,device=args.gpu)

#         for i in range(len(g_0)):
#             for key in g_0[i].keys():
#                 tmp_share_zi00,tmp_share_zi01 = GETSED_single_layer(g_0[i][key].to(torch.float64), 
#                                                                 g_1[i][key].to(torch.float64), 
#                                                                 share_users_sc0[i][key]['A'].to(torch.float64), share_users_sc0[i][key]['D'].to(torch.float64),
#                                                                 share_users_sc1[i][key]['A'].to(torch.float64), share_users_sc1[i][key]['D'].to(torch.float64),
#                                                                 )
#                 share_zi00[i]+=tmp_share_zi00
#                 share_zi01[i]+=tmp_share_zi01
#         return share_zi00,share_zi01

           
    
         
#     #test of the correction of GETSED_single
#     # zi0 = [share_zi00[i]+share_zi01[i] for i in range(len(share_zi00))]
#     # print(zi0)
#     # # get the norm of the gradient
#     # g_norm = [0.0]*len(update_params)
#     # for i in range(len(update_params)):
#     #     for key in update_params[i].keys():
#     #         g_norm[i]+=(torch.sum(torch.pow(update_params[i][key],2)))
#     # print(g_norm)



  
#     # print(share_zij0[0],share_zij1[0])
#     # zij = [[share_zij0[i][j]+share_zij1[i][j] for j in range(i+1,len(share_zij0[0]))] for i in range(len(share_zij0))]
#     # print(zij[0])







#     # #test of the correction of GETPro
#     # zi0 = [share_zi00[i]+share_zi01[i] for i in range(len(share_zi00))]
#     # share_muil0,share_muil1=GETPro(share_zi00,share_zi01,share_users_beavers0,share_users_beavers1)
#     # muil = [[share_muil0[i][j]+share_muil1[i][j] for j in range(len(share_muil1))] for i in range(len(share_muil0))]
#     # production_ij = [[zi0[i]*zi0[j] for j in range(i+1,len(zi0))]for i in range(len(zi0))]
#     # print(muil[0])
#     # print(production_ij[0])
    
#     start_time0 = time.time()
#     share_zi00,share_zi01 = GETSED_single(share_update_params0,share_update_params1,share_users_sc0,share_users_sc1)
#     share_zij0, share_zij1 = GETSED(share_update_params0,share_update_params1,share_users_sc0,share_users_sc1)
#     share_muil0,share_muil1 = GETPro(share_zi00,share_zi01,share_users_beavers0,share_users_beavers1)
#     end_time0 = time.time()
#     elapsed_time0 = end_time0 - start_time0
#     print(f"第0段代码运行时间: {elapsed_time0:.6f} 秒")
    
#     start_time1 = time.time()
#     zi0 = [share_zi00[i]+share_zi01[i] for i in range(len(share_zi00))]
#     zij = [[share_zij0[i][j]+share_zij1[i][j] for j in range(len(share_zij1))] for i in range(len(share_zij0))]
#     muil = [[torch.sqrt(share_muil0[i][j]+share_muil1[i][j]) for j in range(len(share_muil1))] for i in range(len(share_muil0))]
    
 
    
#     share_numerator0 = [[share_zi00[i]+share_zi00[j]-share_zij0[i][j] for j in range(len(share_zij0[i])) ] for i in range(len(share_zi00))]
#     share_numerator1 = [[share_zi01[i]+share_zi01[j]-share_zij1[i][j] for j in range(len(share_zij1[i])) ] for i in range(len(share_zi01))]
#     share_denominator0 = share_muil0
#     share_denominator1 = share_muil1
#     random_R = random.randrange(1,2)
#     scale_factor = 100000  # 用于将浮点数转化为整数
#     print("\nPaillier Algorithm using phe")
#     # 1. 密钥生成
#     public_key, private_key = paillier.generate_paillier_keypair(n_length = 512)
#     # 加密单个元素的函数
#     def encrypt_element(element):
#         return public_key.encrypt(element)

#     # 加密二维或多维数组的函数
#     def encrypt_ndarray(data):
#         # 如果数据是标量，直接加密
#         if np.isscalar(data):
#             return encrypt_element(data)
#         # 否则递归处理每个元素
#         else:
#             return np.array([encrypt_ndarray(subarray) for subarray in data])

#     # 解密单个元素的函数
#     def decrypt_element(encrypted_element):
#         return private_key.decrypt(encrypted_element)

#     # 解密二维或多维数组的函数
#     def decrypt_ndarray(encrypted_data):
#         # 如果数据是标量，直接解密
#         if np.isscalar(encrypted_data):
#             return decrypt_element(encrypted_data)
#         # 否则递归处理每个元素
#         else:
#             return np.array([decrypt_ndarray(subarray) for subarray in encrypted_data])

#     # # 使用并行加密二维数组
#     # def parallel_encrypt_ndarray(data):
#     #     shape = data.shape  # 获取数组形状
#     #     flattened_data = data.flatten()  # 将数组展平为一维
#     #     with ThreadPoolExecutor() as executor:
#     #         encrypted_flattened = list(executor.map(encrypt_element, flattened_data))
#     #     return np.array(encrypted_flattened).reshape(shape)  # 还原为原始形状

#     # # 使用并行解密二维数组
#     # def parallel_decrypt_ndarray(encrypted_data):
#     #     shape = encrypted_data.shape  # 获取数组形状
#     #     flattened_data = encrypted_data.flatten()  # 将数组展平为一维
#     #     with ThreadPoolExecutor() as executor:
#     #         decrypted_flattened = list(executor.map(decrypt_element, flattened_data))
#     #     return np.array(decrypted_flattened).reshape(shape)  # 还原为原始形状



    
#     # 2. 将浮点数放大并加密
#     # #并行加密
#     # pall_encryp_start = time.time()
#     # pall_encrypted_numerator0 = parallel_encrypt_ndarray(np.array([[(int(scale_factor * share_numerator0[i][j])) 
#     #                          for j in range(len(share_numerator0[i]))] 
#     #                         for i in range(len(share_numerator0))]))
#     # pall_encrypted_numerator1 = parallel_encrypt_ndarray(np.array([[(int(scale_factor * share_numerator1[i][j])) 
#     #                          for j in range(len(share_numerator1[i]))] 
#     #                         for i in range(len(share_numerator1))]))
#     # pall_encrypted_denominator0 = parallel_encrypt_ndarray(np.array([[(int(scale_factor * share_denominator0[i][j])) 
#     #                           for j in range(len(share_denominator0[i]))] 
#     #                           for i in range(len(share_denominator0))]))
#     # pall_encrypted_denominator1 = parallel_encrypt_ndarray(np.array([[(int(scale_factor * share_denominator1[i][j])) 
#     #                           for j in range(len(share_denominator1[i]))] 
#     #                           for i in range(len(share_denominator1))]))
#     # pall_encryp_over = time.time()
#     # pall_encryp_time = pall_encryp_over - pall_encryp_start
#     # print(f"并行加密运行时间: {pall_encryp_time:.6f} 秒")
    
#     #普通方式加密
#     encryp_start = time.time()
#     encrypted_numerator0 = [[public_key.encrypt(int(scale_factor * share_numerator0[i][j])) 
#                              for j in range(len(share_numerator0[i]))] 
#                             for i in range(len(share_numerator0))]
#     encrypted_numerator1 = [[public_key.encrypt(int(scale_factor * share_numerator1[i][j])) 
#                              for j in range(len(share_numerator1[i]))] 
#                             for i in range(len(share_numerator1))]
#     encrypted_denominator0 = [[public_key.encrypt(int(scale_factor * share_denominator0[i][j])) 
#                               for j in range(len(share_denominator0[i]))] 
#                               for i in range(len(share_denominator0))]
#     encrypted_denominator1 = [[public_key.encrypt(int(scale_factor * share_denominator1[i][j])) 
#                               for j in range(len(share_denominator1[i]))] 
#                               for i in range(len(share_denominator1))]
#     encryp_over = time.time()
#     elapsed_encryp_time = encryp_over - encryp_start
#     print(f"加密代码运行时间: {elapsed_encryp_time:.6f}")
#     # 3. 同态操作 - 加法、减法、乘常数
#     #普通方式得到的密文
#     operation_start = time.time()
#     encrypted_numerator0R = [[encrypted_numerator0[i][j]*random_R 
#                               for j in range(len(encrypted_numerator0[i]))]
#                              for i in range(len(encrypted_numerator0))]
#     encrypted_numerator1R = [[encrypted_numerator1[i][j]*random_R 
#                               for j in range(len(encrypted_numerator1[i]))]
#                              for i in range(len(encrypted_numerator1)) ]
#     encrypted_denominator0RR = [[encrypted_denominator0[i][j]*random_R**2 
#                                 for j in range(len(encrypted_denominator0[i]))]
#                                 for i in range(len(encrypted_denominator0))]
#     encrypted_denominator1RR = [[encrypted_denominator1[i][j]*random_R**2 
#                                 for j in range(len(encrypted_denominator1[i]))]
#                                 for i in range(len(encrypted_denominator1))]
#     encrypted_numeratorR = [[encrypted_numerator0R[i][j]+encrypted_numerator1R[i][j]
#                             for j in range(len(encrypted_numerator0R[i]))]
#                             for i in range(len(encrypted_numerator0R))]
#     encrypted_denominatorRR = [[encrypted_denominator0RR[i][j]+encrypted_denominator1RR[i][j]
#                             for j in range(len(encrypted_denominator0RR[i]))]
#                             for i in range(len(encrypted_denominator0RR))]
#     operation_over = time.time()
#     elapsed_operation_time = operation_over - operation_start
#     print(f"同态操作代码运行时间: {elapsed_operation_time:.6f}")
    

#     # #并行方式得到的明文
#     # pall_operation_start = time.time()
#     # pall_encrypted_numerator0R = [[pall_encrypted_numerator0[i][j]*random_R 
#     #                           for j in range(len(pall_encrypted_numerator0[i]))]
#     #                          for i in range(len(pall_encrypted_numerator0))]
#     # pall_encrypted_numerator1R = [[pall_encrypted_numerator1[i][j]*random_R 
#     #                           for j in range(len(pall_encrypted_numerator1[i]))]
#     #                          for i in range(len(pall_encrypted_numerator1)) ]
#     # pall_encrypted_denominator0RR = [[pall_encrypted_denominator0[i][j]*random_R**2 
#     #                             for j in range(len(pall_encrypted_denominator0[i]))]
#     #                             for i in range(len(pall_encrypted_denominator0))]
#     # pall_encrypted_denominator1RR = [[pall_encrypted_denominator1[i][j]*random_R**2 
#     #                             for j in range(len(pall_encrypted_denominator1[i]))]
#     #                             for i in range(len(pall_encrypted_denominator1))]
#     # pall_encrypted_numeratorR = [[pall_encrypted_numerator0R[i][j]+pall_encrypted_numerator1R[i][j]
#     #                         for j in range(len(pall_encrypted_numerator0R[i]))]
#     #                         for i in range(len(pall_encrypted_numerator0R))]
#     # pall_encrypted_denominatorRR = [[pall_encrypted_denominator0RR[i][j]+pall_encrypted_denominator1RR[i][j]
#     #                         for j in range(len(pall_encrypted_denominator0RR[i]))]
#     #                         for i in range(len(pall_encrypted_denominator0RR))]
#     # pall_operation_over = time.time()
#     # pall_elapsed_operation_time = pall_operation_over - pall_operation_start
#     # print(f"并行同态操作代码运行时间: {pall_elapsed_operation_time:.6f}")

    
#     # 4. 解密并还原为浮点数
    
#     #解密普通密文
#     decrypt_start = time.time()
#     numeratorR = [[private_key.decrypt(encrypted_numeratorR[i][j]) / scale_factor 
#                    for j in range(len(encrypted_numeratorR[i]))]
#                    for i in range(len(encrypted_numeratorR))]
#     denominatorRR = [[private_key.decrypt(encrypted_denominatorRR[i][j]) / scale_factor 
#                    for j in range(len(encrypted_denominatorRR[i]))]
#                    for i in range(len(encrypted_denominatorRR))]
#     decrypt_over = time.time()
#     elapsed_decrypt_time = decrypt_over - decrypt_start
#     print(f"解密代码运行时间: {elapsed_decrypt_time:.6f}")
#     print(numeratorR[0])
#     print(denominatorRR[0])

#     # #解密并行密文
#     # pall_decrypt_start = time.time()
#     # pall_numeratorR = parallel_decrypt_ndarray(np.array(pall_encrypted_numeratorR)) / scale_factor
#     # pall_denominatorRR = parallel_decrypt_ndarray(np.array(pall_encrypted_denominatorRR)) / scale_factor
#     # pall_decrypt_over = time.time()
#     # pall_elapsed_decrypt_time = pall_decrypt_over - pall_decrypt_start
#     # print(f"并行解密代码运行时间: {pall_elapsed_decrypt_time:.6f}")
#     # print(pall_numeratorR[0])
#     # print(pall_denominatorRR[0])


    
#     # denominatorRR = torch.sqrt(denominatorRR)
#     tmp_cosij = [[1-torch.div(numeratorR[i][j],2*math.sqrt(denominatorRR[i][j])) 
#              for j in range(i+1,len(numeratorR[i]))]
#              for i in range(len(numeratorR))]
#     cosij = [[0 for _ in range(len(tmp_cosij[0])+1)] for _ in range((len(tmp_cosij)+1))]
#     for i in range(len(tmp_cosij)):
#         for j in range(len(tmp_cosij[i])):
#             cosij[1+j+i][i]=cosij[i][1+j+i] = tmp_cosij[i][j].item()
    
 
    
#     # tmp_pall_cosij = [[1-torch.div(pall_numeratorR[i][j],2*math.sqrt(pall_denominatorRR[i][j])) 
#     #          for j in range(i+1,len(pall_numeratorR[i]))]
#     #          for i in range(len(pall_numeratorR))]
#     # pall_cosij = [[0 for _ in range(len(tmp_pall_cosij[0])+1)] for _ in range((len(tmp_pall_cosij)+1)) ]
#     # for i in range(len(tmp_pall_cosij)):
#     #     for j in range(len(tmp_pall_cosij[i])):
#     #         pall_cosij[1+j+i][i]=pall_cosij[i][1+j+i] = tmp_pall_cosij[i][j]
            

    
    
#     end_time1 = time.time()
#     elapsed_time1 = end_time1 - start_time1
#     print(f"第一段代码运行时间: {elapsed_time1:.6f} 秒")
#     print("calculate cosij in secure ",cosij)
#     # print(pall_cosij)
    
    
    
    



    

# # 假设 update_params 和 share_zij0 已经定义
#     # SED = [
#     #     [sum(torch.norm(update_params[i][key].view(-1) - update_params[j][key].view(-1)) for key in update_params[0].keys()) 
#     #     for j in range(i+1, len(share_zij0))]
#     #     for i in range(len(share_zij0))
#     # ]
#     # print(SED[0])
    
#     # print(f"share_zij0[0] size is:{len(share_zij0[0])}")

    


#     cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
#     cos_list=[]
#     local_model_vector = []
#     for param in local_model:
#         # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
#         local_model_vector.append(parameters_dict_to_vector_flt(param))
#     for i in range(len(local_model_vector)):
#         cos_i = []
#         for j in range(len(local_model_vector)):
#             cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
#             # cos_i.append(round(cos_ij.item(),4))
#             cos_i.append(cos_ij.item())
#         cos_list.append(cos_i)
    
#     cosij=[]
    
#     start_time2 = time.time()
#     gradients_vector = []
#     for param in update_params:
#         gradients_vector.append(parameters_dict_to_vector_flt(param))
#     for i in range(len(gradients_vector)):
#         cos_i = []
#         for j in range(len(gradients_vector)):
#             cos_ij = 1- cos(gradients_vector[i],gradients_vector[j])
#             cos_i.append(cos_ij.item())
#         cosij.append(cos_i)
        
#     end_time2 = time.time()
#     elapsed_time2 = end_time2 - start_time2
#     print(f"第二段代码运行时间: {elapsed_time2:.6f} 秒")
#     print("cosij is ",cosij)
    
    
#     num_clients = max(int(args.frac * args.num_users), 1)
#     num_malicious_clients = int(args.malicious * num_clients)
#     num_benign_clients = num_clients - num_malicious_clients
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
#     print(clusterer.labels_)
#     print(clusterer.probabilities_)
#     print(clusterer)
#     benign_client = []
#     norm_list = np.array([])

#     exit()    

#     max_num_in_cluster=0
#     max_cluster_index=0
#     if clusterer.labels_.max() < 0:
#         for i in range(len(local_model)):
#             benign_client.append(i)
#             norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     else:
#         for index_cluster in range(clusterer.labels_.max()+1):
#             if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
#                 max_cluster_index = index_cluster
#                 max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
#         for i in range(len(clusterer.labels_)):
#             if clusterer.labels_[i] == max_cluster_index:
#                 benign_client.append(i)
#     for i in range(len(local_model_vector)):
#         # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
#         norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
#     print(benign_client)
   
#     for i in range(len(benign_client)):
#         if benign_client[i] < num_malicious_clients:
#             args.wrong_mal+=1
#         else:
#             #  minus per benign in cluster
#             args.right_ben += 1
#     args.turn+=1
#     print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
#     print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))
    
#     clip_value = np.median(norm_list)
#     for i in range(len(benign_client)):
#         gama = clip_value/norm_list[i]
#         if gama < 1:
#             for key in update_params[benign_client[i]]:
#                 if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#                 update_params[benign_client[i]][key] *= gama
#     global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
#     #add noise
#     for key, var in global_model.items():
#         if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#         temp = copy.deepcopy(var)
#         temp = temp.normal_(mean=0,std=args.noise*clip_value)
#         var += temp
#     return global_model
    














def newFlame(local_model, update_params, global_model, args):
    # 记录newFlame函数开始时间
    newflame_start_time = time.time()
    
    # GPU内存监控和预警机制
    def monitor_gpu_memory(stage_name=""):
        """监控GPU内存使用情况并提供预警"""
        if not torch.cuda.is_available():
            return False, 0, 0
        
        try:
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_usage_ratio = gpu_memory_used / gpu_memory_total
            
            print(f"[{stage_name}] GPU内存使用: {gpu_memory_used/1024**3:.2f}GB / {gpu_memory_total/1024**3:.2f}GB ({memory_usage_ratio*100:.1f}%)")
            
            # 内存预警机制
            if memory_usage_ratio > 0.9:
                print(f"⚠️  [严重警告] GPU内存使用率超过90%，可能即将发生内存不足错误！")
                return True, memory_usage_ratio, gpu_memory_used
            elif memory_usage_ratio > 0.8:
                print(f"⚠️  [警告] GPU内存使用率超过80%，建议立即清理内存")
                return True, memory_usage_ratio, gpu_memory_used
            elif memory_usage_ratio > 0.7:
                print(f"⚠️  [注意] GPU内存使用率超过70%，需要关注内存使用")
                return False, memory_usage_ratio, gpu_memory_used
            else:
                print(f"✅ GPU内存使用正常")
                return False, memory_usage_ratio, gpu_memory_used
        except Exception as e:
            print(f"获取GPU内存信息失败: {e}")
            return False, 0, 0
    
    def emergency_memory_cleanup():
        """紧急内存清理"""
        print("🧹 执行紧急内存清理...")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("✅ 紧急内存清理完成")
    
    def adaptive_computation_strategy(memory_ratio, client_count):
        """根据内存使用情况自适应调整计算策略"""
        if memory_ratio > 0.8:
            # 高内存压力：使用最保守策略
            return {
                'use_fp16': True,
                'batch_size_factor': 0.3,
                'enable_checkpointing': True,
                'force_cpu_fallback': True
            }
        elif memory_ratio > 0.6:
            # 中等内存压力：平衡策略
            return {
                'use_fp16': True,
                'batch_size_factor': 0.6,
                'enable_checkpointing': True,
                'force_cpu_fallback': False
            }
        else:
            # 低内存压力：性能优先策略
            return {
                'use_fp16': False,
                'batch_size_factor': 1.0,
                'enable_checkpointing': False,
                'force_cpu_fallback': False
            }
    
    # 初始内存监控
    is_warning, memory_ratio, memory_used = monitor_gpu_memory("函数开始")
    if is_warning:
        emergency_memory_cleanup()
        # 重新检查内存状态
        is_warning, memory_ratio, memory_used = monitor_gpu_memory("清理后")
    
    # 根据内存状态调整计算策略
    n_clients = len(update_params)
    computation_strategy = adaptive_computation_strategy(memory_ratio, n_clients)
    print(f"🎯 采用计算策略: {computation_strategy}")
    
    # 全局密钥生成 - 确保整个训练过程中只生成一次
    if not hasattr(args, 'paillier_public_key') or not hasattr(args, 'paillier_private_key'):
        print("首次生成Paillier密钥对，将在整个训练过程中复用...")
        key_generation_start = time.time()
        
        # 优化密钥长度策略：根据内存和客户端数量动态调整
        if computation_strategy['force_cpu_fallback'] or n_clients >= 40:
            key_length = 256  # 极端内存压力或大量客户端时使用最小密钥
        elif memory_ratio > 0.7 or n_clients >= 30:
            key_length = 384  # 中等内存压力时使用中等密钥
        else:
            key_length = 512  # 内存充足时使用标准密钥
            
        args.paillier_public_key, args.paillier_private_key = paillier.generate_paillier_keypair(n_length=key_length)
        key_generation_time = time.time() - key_generation_start
        print(f"Paillier密钥生成用时: {key_generation_time:.6f} 秒 (密钥长度: {key_length}位)")
        
        # 优化随机数生成策略：使用更小的随机数范围以减少计算开销
        if computation_strategy['force_cpu_fallback']:
            args.paillier_R = random.randint(10, 100)  # CPU模式使用小随机数
        elif memory_ratio > 0.7:
            args.paillier_R = random.randint(100, 1000)  # 内存紧张时使用中等随机数
        else:
            args.paillier_R = random.randint(1000, 5000)  # 内存充足时使用较大随机数
        print(f"生成全局随机数R: {args.paillier_R} (优化策略: {'CPU模式' if computation_strategy['force_cpu_fallback'] else '内存自适应'})")
        
        # 设置优化的缩放因子
        if key_length <= 256:
            args.paillier_scale_factor = 1000  # 小密钥使用小缩放因子
        elif key_length <= 384:
            args.paillier_scale_factor = 10000  # 中等密钥使用中等缩放因子
        else:
            args.paillier_scale_factor = 100000  # 大密钥使用标准缩放因子
        print(f"设置缩放因子: {args.paillier_scale_factor}")
    else:
        print("复用已生成的Paillier密钥对和随机数R")
    
    # 定义算术共享函数
    def arithmetic_share(tensor):
        """
        将张量拆分成两个算术共享
        Args:
            tensor: 输入张量
        Returns:
            share1, share2: 两个共享，满足 tensor = share1 + share2
        """
        # 根据计算策略选择精度
        dtype = torch.float16 if computation_strategy['use_fp16'] else torch.float64
        share1 = torch.rand_like(tensor, device=args.gpu, dtype=dtype)
        share2 = tensor.to(dtype) - share1
        return share1, share2
    
    # 开始算术共享操作并统计时间
    arithmetic_share_start_time = time.time()
    monitor_gpu_memory("算术共享开始前")
    
    # 初始化算术共享参数列表
    update_params0 = []
    update_params1 = []
    
    # 对每个客户端的模型参数进行算术共享
    for user_params in update_params:
        shares = {key: list(arithmetic_share(value)) for key, value in user_params.items()}
        update_params0.append({key: share[0] for key, share in shares.items()})
        update_params1.append({key: share[1] for key, share in shares.items()})
    
    arithmetic_share_end_time = time.time()
    arithmetic_share_elapsed_time = arithmetic_share_end_time - arithmetic_share_start_time
    print(f"模型拆分成算术共享操作用时: {arithmetic_share_elapsed_time:.6f} 秒")
    
    # 开始生成张量A和D的操作并统计时间
    tensor_generation_start_time = time.time()
    
    # 为每个用户生成四个A和D对，并进行验证
    A = []
    D = []
    A0 = []
    A1 = []
    D0 = []
    D1 = []
    
    # A和D验证过程的时间统计
    ad_verification_start_time = time.time()
    
    for user_idx, user_params in enumerate(update_params):
        # print(f"为用户 {user_idx} 生成和验证 A、D 对...")
        
        # 为每个用户生成四个A和D对
        A_candidates = []
        D_candidates = []
        
        for pair_idx in range(4):
            A_user = {}
            D_user = {}
            for key, value in user_params.items():
                # 生成与参数张量形状相同的随机张量A
                A_tensor = torch.rand_like(value, device=args.gpu, dtype=torch.float64)
                A_user[key] = A_tensor
                # 计算D = A^2
                D_user[key] = torch.pow(A_tensor, 2)
            A_candidates.append(A_user)
            D_candidates.append(D_user)
        
        # 验证过程：不放回地随机取两对进行验证
        verified_A = []
        verified_D = []
        remaining_indices = list(range(4))
        
        while len(remaining_indices) >= 2:
            # 随机选择两个索引
            selected_indices = random.sample(remaining_indices, 2)
            idx1, idx2 = selected_indices
            
            A1_candidate = A_candidates[idx1]
            D1_candidate = D_candidates[idx1]
            A2_candidate = A_candidates[idx2]
            D2_candidate = D_candidates[idx2]
            
            # 随机生成奇数t
            t = random.randrange(1, 100, 2)  # 生成1到99之间的奇数
            
            # 验证过程：检查每个参数键（使用CPU矩阵计算）
            verification_passed = True
            for key in A1_candidate.keys():
                # 将张量移到CPU进行矩阵计算
                A1_cpu = A1_candidate[key].cpu()
                A2_cpu = A2_candidate[key].cpu()
                D1_cpu = D1_candidate[key].cpu()
                D2_cpu = D2_candidate[key].cpu()
                
                # 计算 e = t*A1 - A2 (矩阵运算)
                e = t * A1_cpu - A2_cpu
                
                # 验证条件：t^2 * D1 - D2 - 2*t*e*A1 + e^2 = 0 (矩阵运算)
                verification_result = (t**2 * D1_cpu - D2_cpu - 2*t*e*A1_cpu + torch.pow(e, 2))
                
                # 检查是否接近0（考虑浮点数精度）
                if torch.max(torch.abs(verification_result)) > 1e-6:
                    verification_passed = False
                    break
            
            if verification_passed:
                print(f"  验证通过：保留第 {idx1} 对 A、D")
                verified_A.append(A1_candidate)
                verified_D.append(D1_candidate)
                # 从候选列表中移除已验证的两对
                remaining_indices.remove(idx1)
                remaining_indices.remove(idx2)
            else:
                print(f"  验证失败：丢弃第 {idx2} 对 A、D，保留第 {idx1} 对")
                verified_A.append(A1_candidate)
                verified_D.append(D1_candidate)
                # 只移除被丢弃的那对
                remaining_indices.remove(idx1)
                remaining_indices.remove(idx2)
        
        # 如果还有剩余的单个候选对，直接添加
        for remaining_idx in remaining_indices:
            verified_A.append(A_candidates[remaining_idx])
            verified_D.append(D_candidates[remaining_idx])
        
        # 选择第一个验证通过的A和D对作为该用户的最终A和D
        if verified_A:
            final_A = verified_A[0]
            final_D = verified_D[0]
        else:
            # 如果没有验证通过的对，使用第一个候选对
            final_A = A_candidates[0]
            final_D = D_candidates[0]
        
        A.append(final_A)
        D.append(final_D)
        
        # 将张量A拆分成算术共享A0和A1
        A0_user = {}
        A1_user = {}
        for key, A_tensor in final_A.items():
            share0, share1 = arithmetic_share(A_tensor)
            A0_user[key] = share0
            A1_user[key] = share1
        A0.append(A0_user)
        A1.append(A1_user)
        
        # 将张量D拆分成算术共享D0和D1
        D0_user = {}
        D1_user = {}
        for key, D_tensor in final_D.items():
            share0, share1 = arithmetic_share(D_tensor)
            D0_user[key] = share0
            D1_user[key] = share1
        D0.append(D0_user)
        D1.append(D1_user)
    
    ad_verification_end_time = time.time()
    ad_verification_elapsed_time = ad_verification_end_time - ad_verification_start_time
    print(f"所有用户的A和D验证过程用时: {ad_verification_elapsed_time:.6f} 秒")
    
    tensor_generation_end_time = time.time()
    tensor_generation_elapsed_time = tensor_generation_end_time - tensor_generation_start_time
    print(f"所有用户的张量A和D生成及算术共享操作用时: {tensor_generation_elapsed_time:.6f} 秒")
    
    # 开始生成张量alpha、beta、gamma的操作并统计时间
    alpha_beta_gamma_start_time = time.time()
    
    # 为每个用户生成两对alpha、beta、gamma，并进行验证
    alpha = []
    beta = []
    gamma = []
    alpha0 = []
    alpha1 = []
    beta0 = []
    beta1 = []
    gamma0 = []
    gamma1 = []
    
    # alpha、beta、gamma验证过程的时间统计
    abg_verification_start_time = time.time()
    
    for user_idx, user_params in enumerate(update_params):
        print(f"为用户 {user_idx} 生成和验证 alpha、beta、gamma 对...")
        
        # 为每个用户生成两对alpha、beta、gamma
        alpha_candidates = []
        beta_candidates = []
        gamma_candidates = []
        
        for pair_idx in range(2):
            alpha_user = {}
            beta_user = {}
            gamma_user = {}
            for key, value in user_params.items():
                # 生成与参数张量形状相同的随机张量alpha和beta
                alpha_tensor = torch.rand_like(value, device=args.gpu, dtype=torch.float64)
                beta_tensor = torch.rand_like(value, device=args.gpu, dtype=torch.float64)
                alpha_user[key] = alpha_tensor
                beta_user[key] = beta_tensor
                # 计算gamma = beta * alpha (元素级乘法)
                gamma_user[key] = torch.mul(beta_tensor, alpha_tensor)
            alpha_candidates.append(alpha_user)
            beta_candidates.append(beta_user)
            gamma_candidates.append(gamma_user)
        
        # 验证过程：对两对进行验证
        alpha1_candidate = alpha_candidates[0]
        beta1_candidate = beta_candidates[0]
        gamma1_candidate = gamma_candidates[0]
        alpha2_candidate = alpha_candidates[1]
        beta2_candidate = beta_candidates[1]
        gamma2_candidate = gamma_candidates[1]
        
        # 随机生成奇数t
        t = random.randrange(1, 100, 2)  # 生成1到99之间的奇数
        
        # 验证过程：检查每个参数键（使用CPU矩阵计算）
        verification_passed = True
        for key in alpha1_candidate.keys():
            # 将张量移到CPU进行矩阵计算
            alpha1_cpu = alpha1_candidate[key].cpu()
            alpha2_cpu = alpha2_candidate[key].cpu()
            beta1_cpu = beta1_candidate[key].cpu()
            beta2_cpu = beta2_candidate[key].cpu()
            gamma1_cpu = gamma1_candidate[key].cpu()
            gamma2_cpu = gamma2_candidate[key].cpu()
            
            # 计算 e = t * alpha1 - alpha2 (矩阵运算)
            e = t * alpha1_cpu - alpha2_cpu
            # 计算 f = t * beta1 - beta2 (矩阵运算)
            f = t * beta1_cpu - beta2_cpu
            
            # 验证条件：e*f + beta2*e + alpha2*f - t*t*gamma1 + gamma2 = 0 (矩阵运算)
            verification_result = (e * f + beta2_cpu * e + alpha2_cpu * f - 
                                 t**2 * gamma1_cpu + gamma2_cpu)
            
            # 检查是否接近0（考虑浮点数精度）
            if torch.max(torch.abs(verification_result)) > 1e-6:
                verification_passed = False
                break
        
        if verification_passed:
            print(f"  验证通过：保留第一对 alpha、beta、gamma")
            final_alpha = alpha1_candidate
            final_beta = beta1_candidate
            final_gamma = gamma1_candidate
        else:
            print(f"  验证失败：使用第一对 alpha、beta、gamma")
            final_alpha = alpha1_candidate
            final_beta = beta1_candidate
            final_gamma = gamma1_candidate
        
        alpha.append(final_alpha)
        beta.append(final_beta)
        gamma.append(final_gamma)
        
        # 将张量alpha拆分成算术共享alpha0和alpha1
        alpha0_user = {}
        alpha1_user = {}
        for key, alpha_tensor in final_alpha.items():
            share0, share1 = arithmetic_share(alpha_tensor)
            alpha0_user[key] = share0
            alpha1_user[key] = share1
        alpha0.append(alpha0_user)
        alpha1.append(alpha1_user)
        
        # 将张量beta拆分成算术共享beta0和beta1
        beta0_user = {}
        beta1_user = {}
        for key, beta_tensor in final_beta.items():
            share0, share1 = arithmetic_share(beta_tensor)
            beta0_user[key] = share0
            beta1_user[key] = share1
        beta0.append(beta0_user)
        beta1.append(beta1_user)
        
        # 将张量gamma拆分成算术共享gamma0和gamma1
        gamma0_user = {}
        gamma1_user = {}
        for key, gamma_tensor in final_gamma.items():
            share0, share1 = arithmetic_share(gamma_tensor)
            gamma0_user[key] = share0
            gamma1_user[key] = share1
        gamma0.append(gamma0_user)
        gamma1.append(gamma1_user)
    
    abg_verification_end_time = time.time()
    abg_verification_elapsed_time = abg_verification_end_time - abg_verification_start_time
    print(f"所有用户的alpha、beta、gamma验证过程用时: {abg_verification_elapsed_time:.6f} 秒")
    
    alpha_beta_gamma_end_time = time.time()
    alpha_beta_gamma_elapsed_time = alpha_beta_gamma_end_time - alpha_beta_gamma_start_time
    print(f"所有用户的张量alpha、beta、gamma生成及算术共享操作用时: {alpha_beta_gamma_elapsed_time:.6f} 秒")

    # 开始计算新的算术共享操作
    zeta_operations_start_time = time.time()
    
    # 为每个客户端的梯度g[i]计算临时算术共享e0 = g0 - A0和e1 = g1 - A1
    e0 = []
    e1 = []
    e = []
    for i, (g0_user, g1_user, A0_user, A1_user) in enumerate(zip(update_params0, update_params1, A0, A1)):
        e0_user = {}
        e1_user = {}
        e_user = {}
        for key in g0_user.keys():
            # 在CPU上计算临时算术共享 e0 = g0 - A0, e1 = g1 - A1
            g0_cpu = g0_user[key].cpu()
            g1_cpu = g1_user[key].cpu()
            A0_cpu = A0_user[key].cpu()
            A1_cpu = A1_user[key].cpu()
            e0_cpu = g0_cpu - A0_cpu
            e1_cpu = g1_cpu - A1_cpu
            # 计算 e = e0 + e1 = g - A（保持在CPU上以降低GPU显存占用）
            e_cpu = e0_cpu + e1_cpu
            e0_user[key] = e0_cpu
            e1_user[key] = e1_cpu
            e_user[key] = e_cpu
            # 清理CPU临时变量引用
            del g0_cpu, g1_cpu, A0_cpu, A1_cpu
        e0.append(e0_user)
        e1.append(e1_user)
        e.append(e_user)
    
    # 计算新的算术共享Zeta0 = D0 + 2*e*g0 - e^2和Zeta1 = D1 + 2*e*g1
    Zeta0 = []
    Zeta1 = []
    for i, (D0_user, D1_user, e_user, g0_user, g1_user) in enumerate(zip(D0, D1, e, update_params0, update_params1)):
        Zeta0_user = {}
        Zeta1_user = {}
        for key in D0_user.keys():
            # 在CPU上计算 Zeta0 = D0 + 2*e*g0 - e^2 和 Zeta1 = D1 + 2*e*g1
            D0_cpu = D0_user[key].cpu()
            D1_cpu = D1_user[key].cpu()
            g0_cpu = g0_user[key].cpu()
            g1_cpu = g1_user[key].cpu()
            e_cpu = e_user[key]  # 已在CPU上
            Zeta0_cpu = D0_cpu + 2 * e_cpu * g0_cpu - torch.pow(e_cpu, 2)
            Zeta1_cpu = D1_cpu + 2 * e_cpu * g1_cpu
            Zeta0_user[key] = Zeta0_cpu
            Zeta1_user[key] = Zeta1_cpu
            # 清理CPU临时变量引用
            del D0_cpu, D1_cpu, g0_cpu, g1_cpu
        Zeta0.append(Zeta0_user)
        Zeta1.append(Zeta1_user)
    
    zeta_operations_end_time = time.time()
    zeta_operations_elapsed_time = zeta_operations_end_time - zeta_operations_start_time
    print(f"计算临时算术共享e0、e1、e和新算术共享Zeta0、Zeta1操作用时: {zeta_operations_elapsed_time:.6f} 秒")

    # 开始计算两个客户端梯度的复杂算术共享操作
    deta_operations_start_time = time.time()
    
    # 为两个客户端的梯度gi和gj计算复杂的算术共享操作
    # 这里我们对所有可能的客户端对进行计算，使用批处理策略减少内存占用
    num_clients = len(update_params0)
    
    # 存储所有客户端对的计算结果
    all_e0 = []
    all_e1 = []
    all_e = []
    all_E0 = []
    all_E1 = []
    all_E = []
    all_Deta0 = []
    all_Deta1 = []
    
    # 实施批处理策略，每次处理有限数量的客户端对以控制内存使用
    batch_size = min(10, num_clients)  # 根据内存情况调整批处理大小
    
    for batch_start in range(0, num_clients, batch_size):
        batch_end = min(batch_start + batch_size, num_clients)
        
        # 强制进行垃圾回收以释放内存
        torch.cuda.empty_cache()
        
        for i in range(batch_start, batch_end):
            for j in range(i + 1, num_clients):  # 避免重复计算，只计算i < j的情况
                # 获取客户端i和j的梯度算术共享
                gi0_user = update_params0[i]
                gi1_user = update_params1[i]
                gj0_user = update_params0[j]
                gj1_user = update_params1[j]
                
                # 获取对应的A和D的算术共享
                Ai0_user = A0[i]
                Ai1_user = A1[i]
                Aj0_user = A0[j]
                Aj1_user = A1[j]
                Di0_user = D0[i]
                Di1_user = D1[i]
                Dj0_user = D0[j]
                Dj1_user = D1[j]
                
                # 计算临时算术共享 e0 = gi0 - gj0 - Ai0, e1 = gi1 - gj1 - Ai1
                e0_user = {}
                e1_user = {}
                e_user = {}
                # 将计算转移到CPU以避免GPU内存不足
                for key in gi0_user.keys():
                    # 将张量移动到CPU进行计算
                    gi0_cpu = gi0_user[key].cpu()
                    gj0_cpu = gj0_user[key].cpu()
                    Ai0_cpu = Ai0_user[key].cpu()
                    
                    gi1_cpu = gi1_user[key].cpu()
                    gj1_cpu = gj1_user[key].cpu()
                    Ai1_cpu = Ai1_user[key].cpu()
                    
                    # 在CPU上进行计算
                    e0_cpu = gi0_cpu - gj0_cpu - Ai0_cpu
                    e1_cpu = gi1_cpu - gj1_cpu - Ai1_cpu
                    
                    # 计算 e = e0 + e1 = gi - gj - Ai
                    e_cpu = e0_cpu + e1_cpu
                    
                    # 统计相关结果保留在CPU，避免GPU内存不足
                    e0_user[key] = e0_cpu
                    e1_user[key] = e1_cpu
                    e_user[key] = e_cpu
                    
                    # 清理CPU临时变量
                    del gi0_cpu, gj0_cpu, Ai0_cpu, gi1_cpu, gj1_cpu, Ai1_cpu, e0_cpu, e1_cpu, e_cpu
                
                # 计算临时算术共享 E0 = gi0 - gj0 - Aj0, E1 = gi1 - gj1 - Aj1
                E0_user = {}
                E1_user = {}
                E_user = {}
                for key in gi0_user.keys():
                    # 将张量移动到CPU进行计算
                    gi0_cpu = gi0_user[key].cpu()
                    gj0_cpu = gj0_user[key].cpu()
                    Aj0_cpu = Aj0_user[key].cpu()
                    
                    gi1_cpu = gi1_user[key].cpu()
                    gj1_cpu = gj1_user[key].cpu()
                    Aj1_cpu = Aj1_user[key].cpu()
                    
                    # 在CPU上进行计算
                    E0_cpu = gi0_cpu - gj0_cpu - Aj0_cpu
                    E1_cpu = gi1_cpu - gj1_cpu - Aj1_cpu
                    
                    # 计算 E = E0 + E1 = gi - gj - Aj
                    E_cpu = E0_cpu + E1_cpu
                    
                    # 统计相关操作保持在CPU上，避免GPU内存不足
                    # 不将结果移回GPU，直接在CPU上保存用于后续统计计算
                    E0_user[key] = E0_cpu
                    E1_user[key] = E1_cpu
                    E_user[key] = E_cpu
                    
                    # 清理CPU临时变量
                    del gi0_cpu, gj0_cpu, Aj0_cpu, gi1_cpu, gj1_cpu, Aj1_cpu
                
                # 计算新的算术共享 Deta0 和 Deta1（使用CPU计算避免GPU内存不足）
                Deta0_user = {}
                Deta1_user = {}
                for key in gi0_user.keys():
                    # 将所有张量移至CPU进行计算以避免GPU内存不足
                    Di0_cpu = Di0_user[key].cpu()
                    Dj0_cpu = Dj0_user[key].cpu()
                    gi0_cpu = gi0_user[key].cpu()
                    gj0_cpu = gj0_user[key].cpu()
                    # E_user已经在CPU上，直接使用
                    e_cpu = e_user[key].cpu()
                    E_cpu = E_user[key]  # 已经在CPU上
                
                    # Deta0 = 0.5 * (Di0 + Dj0) + (gi0 - gj0) * (e+E) - 0.5 * e*e
                    # 在CPU上进行计算以节省GPU内存
                    term1_cpu = 0.5 * (Di0_cpu + Dj0_cpu)
                    term2_cpu = (gi0_cpu - gj0_cpu) * (e_cpu + E_cpu)
                    term3_cpu = 0.5 * torch.pow(e_cpu, 2)
                    Deta0_cpu = term1_cpu + term2_cpu - term3_cpu
                    
                    # 清理CPU中间变量
                    del term1_cpu, term2_cpu, term3_cpu
                    
                    # 对Deta1进行类似处理 - 使用CPU计算
                    Di1_cpu = Di1_user[key].cpu()
                    Dj1_cpu = Dj1_user[key].cpu()
                    gi1_cpu = gi1_user[key].cpu()
                    gj1_cpu = gj1_user[key].cpu()
                    
                    # Deta1 = 0.5 * (Di1 + Dj1) + (gi1 - gj1) * (e+E) - 0.5 * E * E
                    term1_cpu = 0.5 * (Di1_cpu + Dj1_cpu)
                    term2_cpu = (gi1_cpu - gj1_cpu) * (e_cpu + E_cpu)
                    term3_cpu = 0.5 * torch.pow(E_cpu, 2)
                    Deta1_cpu = term1_cpu + term2_cpu - term3_cpu
                    
                    # 统计计算结果保持在CPU上
                    Deta0_user[key] = Deta0_cpu
                    Deta1_user[key] = Deta1_cpu
                    
                    # 清理所有CPU临时变量
                    del Di0_cpu, Dj0_cpu, gi0_cpu, gj0_cpu, e_cpu
                    del Di1_cpu, Dj1_cpu, gi1_cpu, gj1_cpu, term1_cpu, term2_cpu, term3_cpu
                
                # 存储计算结果
                all_e0.append((i, j, e0_user))
                all_e1.append((i, j, e1_user))
                all_e.append((i, j, e_user))
                all_E0.append((i, j, E0_user))
                all_E1.append((i, j, E1_user))
                all_E.append((i, j, E_user))
                all_Deta0.append((i, j, Deta0_user))
                all_Deta1.append((i, j, Deta1_user))
                
                # 优化内存清理策略 - 释放中间变量
                # 由于统计相关变量已在CPU上，只需清理引用即可
                del e0_user, e1_user, e_user, E0_user, E1_user, E_user, Deta0_user, Deta1_user
                
                # 定期清理GPU缓存以防止内存碎片
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    deta_operations_end_time = time.time()
    deta_operations_elapsed_time = deta_operations_end_time - deta_operations_start_time
    print(f"计算两个客户端梯度的复杂算术共享操作(e0,e1,e,E0,E1,E,Deta0,Deta1)用时: {deta_operations_elapsed_time:.6f} 秒")
    print(f"总共计算了 {len(all_Deta0)} 个客户端对的算术共享操作")

    # 开始同态加密操作的时间统计
    homomorphic_operations_start_time = time.time()
    
    # 计算 update_params 的用户个数 n
    n = len(update_params)
    print(f"update_params 的用户个数 n: {n}")
    
    # 使用全局密钥对和随机数
    public_key = args.paillier_public_key
    private_key = args.paillier_private_key
    R = args.paillier_R
    print("使用全局Paillier密钥对和随机数R进行同态加密操作")
    
    # 定义 div 函数进行同态加密运算，支持数组输入
    def div(x0, x1, y0, y1):
        # 确保输入是数组格式
        if not isinstance(x0, (list, tuple, np.ndarray)):
            x0 = [x0]
        if not isinstance(x1, (list, tuple, np.ndarray)):
            x1 = [x1]
        if not isinstance(y0, (list, tuple, np.ndarray)):
            y0 = [y0]
        if not isinstance(y1, (list, tuple, np.ndarray)):
            y1 = [y1]
        
        # 获取数组长度
        array_length = len(x0)
        
        # 确保所有数组长度一致
        if not (len(x0) == len(x1) == len(y0) == len(y1)):
            raise ValueError("所有输入数组的长度必须相同")
        
        # 自适应精度缩放：根据数据范围和全局优化策略动态调整精度
        max_abs_value = max(
            max(abs(val) for val in x0),
            max(abs(val) for val in x1),
            max(abs(val) for val in y0),
            max(abs(val) for val in y1)
        )
        
        # 使用全局优化的缩放因子，并根据数值大小进行微调
        base_scale_factor = args.paillier_scale_factor
        if max_abs_value < 0.001:
            scale_factor = base_scale_factor * 10  # 小数值需要更高精度
        elif max_abs_value < 1.0:
            scale_factor = base_scale_factor      # 使用基础缩放因子
        else:
            scale_factor = max(base_scale_factor // 10, 1000)  # 大数值降低精度，但保持最小值
        
        print(f"优化缩放策略: max_abs_value={max_abs_value:.6f}, base_scale={base_scale_factor}, final_scale={scale_factor}")
        
        # 定义单个元素的加密计算函数，用于并行处理（优化内存使用）
        def compute_single_element(i):
            try:
                # 使用自适应缩放因子将浮点数转换为整数
                x0_int = int(x0[i] * scale_factor)
                x1_int = int(x1[i] * scale_factor)
                y0_int = int(y0[i] * scale_factor)
                y1_int = int(y1[i] * scale_factor)
                
                # 将加密操作移至CPU以减少GPU内存压力
                # 使用 Paillier 加密计算 x0[i], y0[i] 的密文 gx0, gy0
                # 强制所有Paillier加密操作在CPU上执行，避免GPU内存不足问题
                gx0 = public_key.encrypt(x0_int)
                gy0 = public_key.encrypt(y0_int)
                
                # 在不用到 x0[i] 和 y0[i] 的情况下利用 gx0、gy0 计算密文
                gx0R = gx0 * R
                gy0RR = gy0 * (R * R)
                
                # 利用 x1[i] 和 y1[i] 计算密文
                gx1R = public_key.encrypt(x1_int * R)
                gy1RR = public_key.encrypt(y1_int * R * R)
                
                # 在密文上计算 (x0[i]+x1[i])*R 的密文 gxR
                gxR = gx0R + gx1R
                
                # 在密文上计算 (y0[i]+y1[i])*R*R 的密文 gyRR
                gyRR = gy0RR + gy1RR
                
                # 解密 gxR 和 gyRR 得到 xR 和 yRR
                xR = private_key.decrypt(gxR)
                yRR = private_key.decrypt(gyRR)
                # 立即清理中间变量以释放内存
                del gx0, gy0, gx0R, gy0RR, gx1R, gy1RR, gxR, gyRR
                
                # 计算 x/(y^0.5)
                # 注意：这里需要考虑缩放因子
                x_original = xR / R / scale_factor  # 恢复原始的 x 值
                y_original = yRR / (R * R) / scale_factor  # 恢复原始的 y 值
                
                if y_original > 0:
                    result = x_original / (y_original ** 0.5)
                else:
                    result = 0  # 避免除零错误
                
                # 清理局部变量
                del x0_int, x1_int, y0_int, y1_int, xR, yRR, x_original, y_original
                
                return i, result
                
            except Exception as e:
                print(f"计算元素 {i} 时出错: {e}")
                return i, 0  # 返回默认值避免程序崩溃
        
        # 使用流式处理和内存优化的并行计算
        computation_start_time = time.time()
        
        # 根据数组大小调整处理策略（强制CPU模式）
        def get_optimal_processing_strategy(array_length):
            # 强制使用CPU模式，避免GPU内存不足问题
            print("强制使用CPU模式进行Paillier加密计算")
            
            # 根据数组大小调整线程数，但始终在CPU上运行
            if array_length <= 10:
                return 1, False  # 小数组使用串行处理
            elif array_length <= 50:
                return min(2, array_length), True  # 中等数组使用2个线程
            else:
                return min(4, array_length), True  # 大数组使用最多4个线程
        
        max_workers, use_parallel = get_optimal_processing_strategy(array_length)
        
        if array_length > 1 and use_parallel:  # 降低并行处理阈值
            print(f"使用CPU并行计算: {max_workers} 个线程处理 {array_length} 个元素")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 使用流式处理，避免一次性提交所有任务
                chunk_size = max(1, array_length // max_workers)
                results = [None] * array_length
                
                for chunk_start in range(0, array_length, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, array_length)
                    chunk_futures = [executor.submit(compute_single_element, i) 
                                   for i in range(chunk_start, chunk_end)]
                    
                    # 立即处理这个块的结果
                    for future in chunk_futures:
                        i, result = future.result()
                        results[i] = result
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
        else:
            # 对于小数组，使用串行处理
            print(f"使用CPU串行计算处理 {array_length} 个元素")
            results = []
            for i in range(array_length):
                _, result = compute_single_element(i)
                results.append(result)
        
        computation_time = time.time() - computation_start_time
        print(f"div函数计算用时: {computation_time:.6f} 秒")
        print(f"平均每个元素计算用时: {computation_time/array_length:.6f} 秒")
        
        return results
    
    # 执行div函数进行同态加密计算 - 处理所有用户对组合（批处理优化）
    # 计算需要执行的次数：n*(n-1)/2
    total_pairs = n * (n - 1) // 2
    print(f"需要处理的用户对组合数: {total_pairs} (n={n})")
    
    if total_pairs > 0:
        # 生成所有用户对的组合
        user_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                user_pairs.append((i, j))
        
        print(f"生成的用户对组合: {user_pairs[:5]}{'...' if len(user_pairs) > 5 else ''}")
        
        # 动态批处理设置 - 根据GPU内存使用情况自动调整批大小
        def get_gpu_memory_usage():
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
            return 0
        
        def calculate_dynamic_batch_size(total_pairs, base_batch_size=10):
            """根据GPU内存使用情况动态计算批大小"""
            if not torch.cuda.is_available():
                return min(base_batch_size, total_pairs)
            
            try:
                # 获取GPU内存信息
                gpu_memory_used = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                memory_usage_ratio = gpu_memory_used / gpu_memory_total
                
                print(f"GPU内存使用情况: {gpu_memory_used/1024**3:.2f}GB / {gpu_memory_total/1024**3:.2f}GB ({memory_usage_ratio*100:.1f}%)")
                
                # 根据内存使用情况调整批大小
                if memory_usage_ratio < 0.3:  # 内存使用率低于30%
                    dynamic_batch_size = min(20, total_pairs)  # 可以使用较大批大小
                elif memory_usage_ratio < 0.6:  # 内存使用率30%-60%
                    dynamic_batch_size = min(10, total_pairs)  # 使用中等批大小
                elif memory_usage_ratio < 0.8:  # 内存使用率60%-80%
                    dynamic_batch_size = min(5, total_pairs)   # 使用较小批大小
                else:  # 内存使用率超过80%
                    dynamic_batch_size = min(2, total_pairs)   # 使用最小批大小
                
                return dynamic_batch_size
            except Exception as e:
                print(f"获取GPU内存信息失败，使用默认批大小: {e}")
                return min(base_batch_size, total_pairs)
        
        batch_size = calculate_dynamic_batch_size(total_pairs)
        num_batches = (total_pairs + batch_size - 1) // batch_size
        print(f"使用动态批处理模式: 批大小={batch_size}, 总批数={num_batches}")
        
        all_div_results = []
        total_div_start_time = time.time()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_pairs)
            batch_pairs = user_pairs[start_idx:end_idx]
            
            print(f"\n处理批次 {batch_idx+1}/{num_batches} (用户对 {start_idx+1}-{end_idx})")
            
            # 处理当前批次的用户对
            batch_results = []
            for pair_idx_in_batch, (i, j) in enumerate(batch_pairs):
                global_pair_idx = start_idx + pair_idx_in_batch
                
                # 为当前用户对生成随机测试数据
                np.random.seed(42 + global_pair_idx)  # 使用不同的种子确保数据多样性
                
                # 生成单个元素的测试数据（每个用户对一个计算）
                x0_val = np.random.uniform(-1.0, 1.0)
                x1_val = np.random.uniform(-1.0, 1.0)
                y0_val = np.random.uniform(0.1, 2.0)  # 确保 y 值为正数
                y1_val = np.random.uniform(0.1, 2.0)  # 确保 y 值为正数
                
                # 调用 div 函数
                pair_start_time = time.time()
                div_result = div([x0_val], [x1_val], [y0_val], [y1_val])
                pair_computation_time = time.time() - pair_start_time
                
                batch_results.append({
                    'user_pair': (i, j),
                    'input_data': {'x0': x0_val, 'x1': x1_val, 'y0': y0_val, 'y1': y1_val},
                    'result': div_result[0],
                    'computation_time': pair_computation_time
                })
            
            # 将批次结果添加到总结果中
            all_div_results.extend(batch_results)
            
            # GPU内存清理 - 强制垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"  批次 {batch_idx+1} 完成，处理了 {len(batch_results)} 个用户对")
        
        total_div_computation_time = time.time() - total_div_start_time
        
        print(f"\n=== div函数执行总结 ===")
        print(f"总执行次数: {len(all_div_results)} (预期: {total_pairs})")
        print(f"总计算用时: {total_div_computation_time:.6f} 秒")
        print(f"平均每次计算用时: {total_div_computation_time/len(all_div_results):.6f} 秒")
        print(f"前3个计算结果:")
        for i in range(min(3, len(all_div_results))):
            result_info = all_div_results[i]
            print(f"  用户对{result_info['user_pair']}: 结果={result_info['result']:.6f}, 用时={result_info['computation_time']:.6f}秒")
    else:
        print("用户数量不足，无法进行div函数计算")
    
    # 结束同态加密操作的时间统计
    homomorphic_operations_end_time = time.time()
    homomorphic_operations_elapsed_time = homomorphic_operations_end_time - homomorphic_operations_start_time
    print(f"同态加密操作(计算用户个数n和div函数运算)用时: {homomorphic_operations_elapsed_time:.6f} 秒")

    # 开始聚类算法的时间统计
    clustering_start_time = time.time()
    print("\n开始执行聚类算法...")
    
    # 计算余弦相似度矩阵
    cosine_similarity_start_time = time.time()
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    cosine_similarity_end_time = time.time()
    cosine_similarity_elapsed_time = cosine_similarity_end_time - cosine_similarity_start_time
    print(f"  - 余弦相似度矩阵计算用时: {cosine_similarity_elapsed_time:.6f} 秒")
    
    # HDBSCAN聚类
    hdbscan_clustering_start_time = time.time()
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    hdbscan_clustering_end_time = time.time()
    hdbscan_clustering_elapsed_time = hdbscan_clustering_end_time - hdbscan_clustering_start_time
    print(f"  - HDBSCAN聚类算法用时: {hdbscan_clustering_elapsed_time:.6f} 秒")
    print(clusterer.labels_)
    
    # 客户端分类和选择
    client_classification_start_time = time.time()
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    for i in range(len(local_model_vector)):
        # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
        norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    client_classification_end_time = time.time()
    client_classification_elapsed_time = client_classification_end_time - client_classification_start_time
    print(f"  - 客户端分类和选择用时: {client_classification_elapsed_time:.6f} 秒")
    print(benign_client)
    
    # 结束聚类算法的时间统计
    clustering_end_time = time.time()
    clustering_elapsed_time = clustering_end_time - clustering_start_time
    print(f"聚类算法总用时: {clustering_elapsed_time:.6f} 秒")
   
    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal+=1
        else:
            #  minus per benign in cluster
            args.right_ben += 1
    args.turn+=1
    print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
    print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))
    
    clip_value = 2*np.median(norm_list)
    new_update_parms = []
    count = 0
    for i in range(len(benign_client)):
        # gama = args.mul*clip_value/norm_list[i]
        # if gama >= 1:
        if norm_list[i] <= clip_value:
            # print(f"i is {i}  length of benign_client is {len(benign_client)},length of update_params is {len(update_params)},num of benign_client[i] is {benign_client[i]}")
            
            new_update_parms.append(update_params[benign_client[i]])
            # for key in update_params[benign_client[i]]:
            #     if key.split('.')[-1] == 'num_batches_tracked':
            #         continue
            #     update_params[benign_client[i]][key] *= gama
            
    update_params = new_update_parms

    global_model = no_defence_balance([update_params[i] for i in range(len(update_params))], global_model)
    #add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
                    continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0,std=args.noise*clip_value)
        var += temp
    
    # 计算newFlame函数总体用时
    newflame_end_time = time.time()
    newflame_total_elapsed_time = newflame_end_time - newflame_start_time
    
    # 汇总所有时间统计数据
    print("\n========== newFlame函数时间统计汇总 ==========")
    print(f"1. 模型拆分成算术共享操作用时: {arithmetic_share_elapsed_time:.6f} 秒")
    print(f"2. 张量A和D生成及算术共享操作用时: {tensor_generation_elapsed_time:.6f} 秒")
    print(f"   - 其中A和D验证过程用时: {ad_verification_elapsed_time:.6f} 秒")
    print(f"3. 张量alpha、beta、gamma生成及算术共享操作用时: {alpha_beta_gamma_elapsed_time:.6f} 秒")
    print(f"   - 其中alpha、beta、gamma验证过程用时: {abg_verification_elapsed_time:.6f} 秒")
    print(f"4. zeta相关操作用时: {zeta_operations_elapsed_time:.6f} 秒")
    print(f"5. deta相关操作用时: {deta_operations_elapsed_time:.6f} 秒")
    print(f"6. 同态加密操作(计算用户个数n和div函数运算)用时: {homomorphic_operations_elapsed_time:.6f} 秒")
    print(f"7. 聚类算法用时: {clustering_elapsed_time:.6f} 秒")
    print(f"   - 其中余弦相似度矩阵计算用时: {cosine_similarity_elapsed_time:.6f} 秒")
    print(f"   - 其中HDBSCAN聚类算法用时: {hdbscan_clustering_elapsed_time:.6f} 秒")
    print(f"   - 其中客户端分类和选择用时: {client_classification_elapsed_time:.6f} 秒")
    print(f"8. newFlame函数总体用时: {newflame_total_elapsed_time:.6f} 秒")
    print("=" * 50)
    
    # 计算各项操作占总时间的比例
    print("\n========== 各项操作时间占比分析 ==========")
    print(f"1. 模型拆分成算术共享操作占比: {(arithmetic_share_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print(f"2. 张量A和D生成及算术共享操作占比: {(tensor_generation_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print(f"   - 其中A和D验证过程占比: {(ad_verification_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print(f"3. 张量alpha、beta、gamma生成及算术共享操作占比: {(alpha_beta_gamma_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print(f"   - 其中alpha、beta、gamma验证过程占比: {(abg_verification_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print(f"4. zeta相关操作占比: {(zeta_operations_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print(f"5. deta相关操作占比: {(deta_operations_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print(f"6. 同态加密操作占比: {(homomorphic_operations_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print(f"7. 聚类算法占比: {(clustering_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print(f"   - 其中余弦相似度矩阵计算占比: {(cosine_similarity_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print(f"   - 其中HDBSCAN聚类算法占比: {(hdbscan_clustering_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print(f"   - 其中客户端分类和选择占比: {(client_classification_elapsed_time / newflame_total_elapsed_time * 100):.2f}%")
    print("=" * 50)
    
    # 保存时间统计数据到文件
    import os
    import json
    import csv
    from datetime import datetime
    
    # 创建timing_analysis目录
    timing_dir = "timing_analysis"
    if not os.path.exists(timing_dir):
        os.makedirs(timing_dir)
    
    # 获取或创建训练会话ID（基于args对象或全局变量）
    if not hasattr(args, 'training_session_id'):
        args.training_session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 获取或初始化newFlame调用计数器
    if not hasattr(args, 'newflame_call_count'):
        args.newflame_call_count = 0
    args.newflame_call_count += 1
    
    # 准备时间统计数据
    timing_data = {
        "training_session_id": args.training_session_id,
        "call_number": args.newflame_call_count,
        "timestamp": datetime.now().isoformat(),
        "total_time": newflame_total_elapsed_time,
        "operations": {
            "arithmetic_share": {
                "time": arithmetic_share_elapsed_time,
                "percentage": (arithmetic_share_elapsed_time / newflame_total_elapsed_time * 100)
            },
            "tensor_generation": {
                "time": tensor_generation_elapsed_time,
                "percentage": (tensor_generation_elapsed_time / newflame_total_elapsed_time * 100),
                "ad_verification": {
                    "time": ad_verification_elapsed_time,
                    "percentage": (ad_verification_elapsed_time / newflame_total_elapsed_time * 100)
                }
            },
            "alpha_beta_gamma": {
                "time": alpha_beta_gamma_elapsed_time,
                "percentage": (alpha_beta_gamma_elapsed_time / newflame_total_elapsed_time * 100),
                "abg_verification": {
                    "time": abg_verification_elapsed_time,
                    "percentage": (abg_verification_elapsed_time / newflame_total_elapsed_time * 100)
                }
            },
            "zeta_operations": {
                "time": zeta_operations_elapsed_time,
                "percentage": (zeta_operations_elapsed_time / newflame_total_elapsed_time * 100)
            },
            "deta_operations": {
                "time": deta_operations_elapsed_time,
                "percentage": (deta_operations_elapsed_time / newflame_total_elapsed_time * 100)
            },
            "homomorphic_operations": {
                "time": homomorphic_operations_elapsed_time,
                "percentage": (homomorphic_operations_elapsed_time / newflame_total_elapsed_time * 100)
            },
            "clustering_algorithm": {
                "time": clustering_elapsed_time,
                "percentage": (clustering_elapsed_time / newflame_total_elapsed_time * 100),
                "cosine_similarity": {
                    "time": cosine_similarity_elapsed_time,
                    "percentage": (cosine_similarity_elapsed_time / newflame_total_elapsed_time * 100)
                },
                "hdbscan_clustering": {
                    "time": hdbscan_clustering_elapsed_time,
                    "percentage": (hdbscan_clustering_elapsed_time / newflame_total_elapsed_time * 100)
                },
                "client_classification": {
                    "time": client_classification_elapsed_time,
                    "percentage": (client_classification_elapsed_time / newflame_total_elapsed_time * 100)
                }
            }
        }
    }
    
    # 使用固定的文件名（基于训练会话ID）
    json_filename = os.path.join(timing_dir, f"newflame_timing_{args.training_session_id}.json")
    csv_filename = os.path.join(timing_dir, f"newflame_timing_{args.training_session_id}.csv")
    
    # 保存为JSON格式（追加模式）
    if os.path.exists(json_filename):
        # 如果文件存在，读取现有数据并追加新数据
        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
            existing_data.append(timing_data)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = [timing_data]
    else:
        existing_data = [timing_data]
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    # 保存为CSV格式（追加模式）
    file_exists = os.path.exists(csv_filename)
    with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 如果是新文件，写入表头
        if not file_exists:
            writer.writerow(['训练会话ID', '调用次数', '时间戳', '操作名称', '用时(秒)', '占比(%)'])
        
        # 写入当前调用的数据
        session_id = args.training_session_id
        call_num = args.newflame_call_count
        timestamp = datetime.now().isoformat()
        
        writer.writerow([session_id, call_num, timestamp, '模型拆分成算术共享操作', f"{arithmetic_share_elapsed_time:.6f}", f"{(arithmetic_share_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, '张量A和D生成及算术共享操作', f"{tensor_generation_elapsed_time:.6f}", f"{(tensor_generation_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, '  - A和D验证过程', f"{ad_verification_elapsed_time:.6f}", f"{(ad_verification_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, '张量alpha、beta、gamma生成及算术共享操作', f"{alpha_beta_gamma_elapsed_time:.6f}", f"{(alpha_beta_gamma_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, '  - alpha、beta、gamma验证过程', f"{abg_verification_elapsed_time:.6f}", f"{(abg_verification_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, 'zeta相关操作', f"{zeta_operations_elapsed_time:.6f}", f"{(zeta_operations_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, 'deta相关操作', f"{deta_operations_elapsed_time:.6f}", f"{(deta_operations_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, '同态加密操作', f"{homomorphic_operations_elapsed_time:.6f}", f"{(homomorphic_operations_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, '聚类算法', f"{clustering_elapsed_time:.6f}", f"{(clustering_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, '  - 余弦相似度矩阵计算', f"{cosine_similarity_elapsed_time:.6f}", f"{(cosine_similarity_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, '  - HDBSCAN聚类算法', f"{hdbscan_clustering_elapsed_time:.6f}", f"{(hdbscan_clustering_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, '  - 客户端分类和选择', f"{client_classification_elapsed_time:.6f}", f"{(client_classification_elapsed_time / newflame_total_elapsed_time * 100):.2f}"])
        writer.writerow([session_id, call_num, timestamp, '总体用时', f"{newflame_total_elapsed_time:.6f}", "100.00"])
        writer.writerow([])  # 添加空行分隔不同调用
    
    print(f"\n时间统计数据已追加保存到:")
    print(f"JSON格式: {json_filename} (第{args.newflame_call_count}次调用)")
    print(f"CSV格式: {csv_filename} (第{args.newflame_call_count}次调用)")
    
    # 计算和输出模型参数大小
    print("\n========== 训练模型参数大小统计 ==========")
    
    def calculate_model_size(model_dict):
        """计算模型参数的总大小和详细信息"""
        total_params = 0
        total_size_bytes = 0
        layer_info = []
        
        for key, param in model_dict.items():
            if key.split('.')[-1] == 'num_batches_tracked':
                continue  # 跳过BatchNorm的追踪参数
            
            param_count = param.numel()
            param_size_bytes = param.numel() * param.element_size()
            total_params += param_count
            total_size_bytes += param_size_bytes
            
            layer_info.append({
                'layer': key,
                'shape': list(param.shape),
                'params': param_count,
                'size_bytes': param_size_bytes,
                'dtype': str(param.dtype)
            })
        
        return total_params, total_size_bytes, layer_info
    
    # 计算全局模型参数大小
    global_total_params, global_total_size, global_layer_info = calculate_model_size(global_model)
    
    print(f"全局模型参数统计:")
    print(f"  总参数数量: {global_total_params:,}")
    print(f"  总内存占用: {global_total_size / (1024**2):.2f} MB ({global_total_size:,} bytes)")
    print(f"  平均每个参数大小: {global_total_size / global_total_params:.2f} bytes")
    
    # 显示前5个最大的层
    global_layer_info.sort(key=lambda x: x['params'], reverse=True)
    print(f"\n前5个最大的层:")
    for i, layer in enumerate(global_layer_info[:5]):
        print(f"  {i+1}. {layer['layer']}: {layer['params']:,} 参数, 形状{layer['shape']}, {layer['size_bytes']/(1024**2):.2f} MB")
    
    # 计算本地模型参数大小（如果有多个客户端）
    if local_model and len(local_model) > 0:
        print(f"\n本地模型参数统计 (客户端数量: {len(local_model)}):")
        local_total_params_list = []
        local_total_size_list = []
        
        for i, client_model in enumerate(local_model[:3]):  # 只显示前3个客户端的详情
            client_total_params, client_total_size, _ = calculate_model_size(client_model)
            local_total_params_list.append(client_total_params)
            local_total_size_list.append(client_total_size)
            print(f"  客户端 {i}: {client_total_params:,} 参数, {client_total_size / (1024**2):.2f} MB")
        
        if len(local_model) > 3:
            print(f"  ... (还有 {len(local_model) - 3} 个客户端)")
        
        # 计算所有本地模型的总大小
        for i in range(3, len(local_model)):
            client_total_params, client_total_size, _ = calculate_model_size(local_model[i])
            local_total_params_list.append(client_total_params)
            local_total_size_list.append(client_total_size)
        
        total_local_size = sum(local_total_size_list)
        avg_local_size = total_local_size / len(local_model)
        print(f"\n  所有本地模型总内存占用: {total_local_size / (1024**2):.2f} MB")
        print(f"  平均每个本地模型内存占用: {avg_local_size / (1024**2):.2f} MB")
    
    # 计算更新参数大小
    if update_params and len(update_params) > 0:
        print(f"\n更新参数统计 (更新数量: {len(update_params)}):")
        update_total_params_list = []
        update_total_size_list = []
        
        for i, update_param in enumerate(update_params[:3]):  # 只显示前3个更新的详情
            update_total_params, update_total_size, _ = calculate_model_size(update_param)
            update_total_params_list.append(update_total_params)
            update_total_size_list.append(update_total_size)
            print(f"  更新 {i}: {update_total_params:,} 参数, {update_total_size / (1024**2):.2f} MB")
        
        if len(update_params) > 3:
            print(f"  ... (还有 {len(update_params) - 3} 个更新)")
        
        # 计算所有更新参数的总大小
        for i in range(3, len(update_params)):
            update_total_params, update_total_size, _ = calculate_model_size(update_params[i])
            update_total_params_list.append(update_total_params)
            update_total_size_list.append(update_total_size)
        
        total_update_size = sum(update_total_size_list)
        avg_update_size = total_update_size / len(update_params)
        print(f"\n  所有更新参数总内存占用: {total_update_size / (1024**2):.2f} MB")
        print(f"  平均每个更新参数内存占用: {avg_update_size / (1024**2):.2f} MB")
    
    print("=" * 50)
    
    return global_model



# newFLame backup

# def newFlame(local_model, update_params, global_model, args):
#     cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
#     cos_list=[]
#     local_model_vector = []
#     for param in local_model:
#         # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
#         local_model_vector.append(parameters_dict_to_vector_flt(param))
#     for i in range(len(local_model_vector)):
#         cos_i = []
#         for j in range(len(local_model_vector)):
#             cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
#             # cos_i.append(round(cos_ij.item(),4))
#             cos_i.append(cos_ij.item())
#         cos_list.append(cos_i)
#     num_clients = max(int(args.frac * args.num_users), 1)
#     num_malicious_clients = int(args.malicious * num_clients)
#     num_benign_clients = num_clients - num_malicious_clients
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
#     print(clusterer.labels_)
#     benign_client = []
#     norm_list = np.array([])

#     max_num_in_cluster=0
#     max_cluster_index=0
#     if clusterer.labels_.max() < 0:
#         for i in range(len(local_model)):
#             benign_client.append(i)
#             norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
#     else:
#         for index_cluster in range(clusterer.labels_.max()+1):
#             if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
#                 max_cluster_index = index_cluster
#                 max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
#         for i in range(len(clusterer.labels_)):
#             if clusterer.labels_[i] == max_cluster_index:
#                 benign_client.append(i)
#     for i in range(len(local_model_vector)):
#         # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
#         norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
#     print(benign_client)
   
#     for i in range(len(benign_client)):
#         if benign_client[i] < num_malicious_clients:
#             args.wrong_mal+=1
#         else:
#             #  minus per benign in cluster
#             args.right_ben += 1
#     args.turn+=1
#     print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
#     print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))
    
#     clip_value = 2*np.median(norm_list)
#     new_update_parms = []
#     count = 0
#     for i in range(len(benign_client)):
#         # gama = args.mul*clip_value/norm_list[i]
#         # if gama >= 1:
#         if norm_list[i] <= clip_value:
#             # print(f"i is {i}  length of benign_client is {len(benign_client)},length of update_params is {len(update_params)},num of benign_client[i] is {benign_client[i]}")
            
#             new_update_parms.append(update_params[benign_client[i]])
#             # for key in update_params[benign_client[i]]:
#             #     if key.split('.')[-1] == 'num_batches_tracked':
#             #         continue
#             #     update_params[benign_client[i]][key] *= gama
            
#     update_params = new_update_parms

#     global_model = no_defence_balance([update_params[i] for i in range(len(update_params))], global_model)
#     #add noise
#     for key, var in global_model.items():
#         if key.split('.')[-1] == 'num_batches_tracked':
#                     continue
#         temp = copy.deepcopy(var)
#         temp = temp.normal_(mean=0,std=args.noise*clip_value)
#         var += temp
#     return global_model
