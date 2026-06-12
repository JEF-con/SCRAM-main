import sys

sys.path.append('../')

from random import random
from models.test import test_img
from models.Nets import ResNet18, vgg19_bn, vgg19, get_model
from torch.utils.data import DataLoader, Dataset
from utils.options import args_parser

import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch import nn, autograd
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import matplotlib
import os
import random
import time
import math
import heapq
import argparse
import pickle
import torchvision
import cv2

"""
Edge-Case 攻击实现说明（参考 hsq/Multi-metrics-main）

目标：在联邦学习的恶意客户端本地训练中，模拟 Edge-Case/OOD 投毒：
- 对 CIFAR10，将收集的 OOD 图像（如 Southwest 航空飞机图）混入训练集；
- Edge-Case：攻击者握有少量 OOD 样本，并将其标签翻转为目标类（如 truck=9）；
- Normal/Almost-Edge：少量正常客户端也持有 OOD 样本，且保留真实标签（如 airplane=0）。

本实现为 Attacker 侧的简化版本：
- 在恶意训练时构造一个批次级混合数据集，将 OOD 图像以一定比例拼接进来；
- Edge-Case 模式下对 OOD 标签进行翻转；
- 可选支持 `args.attack_case in {edge-case, normal-case, almost-edge-case}` 和
  `args.poison_type == southwest`；
"""


def benign_train(model, dataset, args):
    train_loader = DataLoader(dataset, batch_size = 64, shuffle=True)
    learning_rate = 0.1
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.5)

    for images, labels in train_loader:
        images, labels = images.to(args.device), labels.to(args.device)
        model.zero_grad()
        log_probs = model(images)
        loss = error(log_probs, labels)
        loss.backward()
        optimizer.step()
def malicious_train(model, dataset, args):
    """标准像素触发器恶意训练（用于 edge-case 回退）。

    返回 `(state_dict, avg_loss)`，与主流程接口保持一致。
    """
    model.train()
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=getattr(args, 'lr', 0.1), momentum=getattr(args, 'momentum', 0.5)
    )
    epoch_loss = []
    train_loader = DataLoader(dataset, batch_size=getattr(args, 'local_bs', 64), shuffle=True)
    for iter in range(getattr(args, 'local_ep', 1)):
        batch_loss = []
        for images, labels in train_loader:
            bad_data, bad_label = copy.deepcopy(images), copy.deepcopy(labels)
            for xx in range(len(bad_data)):
                bad_label[xx] = args.attack_label
                bad_data[xx] = add_trigger(args, bad_data[xx])
            images = torch.cat((images, bad_data), dim=0)
            labels = torch.cat((labels, bad_label))
            images, labels = images.to(args.device), labels.to(args.device)
            model.zero_grad()
            log_probs = model(images)
            loss = error(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return model.state_dict(), (sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0)

def _load_southwest_ood(args, attack_case=None):
    """加载 Southwest OOD 图像与标签，路径与 Multi-metrics-main 保持一致。

    返回: (train_images_np, train_labels_np), (test_images_np, test_labels_np)
    若文件缺失或不支持的数据集，返回 (None, None)。
    """
    try:
        base = os.path.join(os.path.dirname(__file__), '..', 'hsq', 'Multi-metrics-main', 'saved_datasets')
        if attack_case is None:
            attack_case = getattr(args, 'attack_case', 'edge-case')
        if attack_case == 'edge-case':
            train_pkl = os.path.join(base, 'southwest_images_new_train.pkl')
            test_pkl = os.path.join(base, 'southwest_images_new_test.pkl')
        elif attack_case in ('normal-case', 'almost-edge-case'):
            train_pkl = os.path.join(base, 'southwest_images_adv_p_percent_edge_case.pkl')
            test_pkl = os.path.join(base, 'southwest_images_p_percent_edge_case_test.pkl')
        else:
            return None, None

        with open(train_pkl, 'rb') as f:
            train_np = pickle.load(f)
        with open(test_pkl, 'rb') as f:
            test_np = pickle.load(f)

        # 标签策略：Edge-Case 翻转到目标类（可配置 edge_target_label）；Normal/Almost-Edge 用真实标签 airplane=0
        target_label = int(getattr(args, 'edge_target_label', 9))
        if attack_case == 'edge-case':
            train_lbl = target_label * np.ones((train_np.shape[0],), dtype=int)
            test_lbl = target_label * np.ones((test_np.shape[0],), dtype=int)
        else:
            train_lbl = 0 * np.ones((train_np.shape[0],), dtype=int)
            test_lbl = 0 * np.ones((test_np.shape[0],), dtype=int)
        return (train_np, train_lbl), (test_np, test_lbl)
    except Exception:
        return None, None

def _sample_and_mix_edgecase(images, labels, args,
                             ood_train_np, ood_train_lbl,
                             clean_sample_M=400, poison_sample_N=None):
    """对一个批次进行 Edge-Case/OOD 混合。
    - 从当前批次复制为 clean 子集（这里直接使用传入批次，不做 M 下采样）。
    - 采样 N 个 OOD 图像，按策略决定其标签。
    - 返回混合后的张量与标签张量。
    """
    device = args.device
    # 将 OOD numpy 转换为张量并归一化到与 CIFAR10 transform 相容的范围
    # 这里假设上游数据已按 0–255 存储；我们用 ToTensor 后再 normalize 的近似：先缩放到 [0,1] 再标准化
    # 为简化，直接用 [0,1] 张量且不额外 normalize（与现有 benign/malicious_train 一致）
    num_ood = ood_train_np.shape[0]
    if poison_sample_N is None:
        poison_sample_N = int(getattr(args, 'edge_poison_n', 32))
    # 按 poison_frac 缩放注入强度
    frac = float(getattr(args, 'poison_frac', 1.0))
    poison_sample_N = max(1, int(poison_sample_N * max(0.0, min(frac, 1.0))))
    select = np.random.choice(num_ood, size=min(poison_sample_N, num_ood), replace=False)
    ood_np_sel = ood_train_np[select]
    ood_lbl_sel = ood_train_lbl[select]

    # 转为 torch，形状 [N, H, W, C] -> [N, C, H, W]
    ood_tensor = torch.tensor(ood_np_sel, dtype=torch.float32)
    if ood_tensor.ndim == 3:
        # 处理灰度情况
        ood_tensor = ood_tensor.unsqueeze(-1)
    ood_tensor = ood_tensor.permute(0, 3, 1, 2) / 255.0
    # 归一化到与 CIFAR10 Transform 一致：Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    if getattr(args, 'dataset', 'cifar') == 'cifar':
        ood_tensor = (ood_tensor - 0.5) / 0.5
    ood_labels = torch.tensor(ood_lbl_sel, dtype=torch.long)

    # 拼接：clean + OOD
    mixed_images = torch.cat([images, ood_tensor.to(images.device)], dim=0)
    mixed_labels = torch.cat([labels, ood_labels.to(labels.device)], dim=0)
    return mixed_images, mixed_labels

def train_malicious_edgecase(model, dataset, args):
    """恶意训练：Edge-Case/OOD 变体。

    返回 `(state_dict, avg_loss)`；当 OOD 资源不可用或数据集不匹配时回退到 `malicious_train`。
    前置假设：`args.dataset == 'cifar' and args.poison_type == 'southwest'`。
    """
    if getattr(args, 'dataset', 'cifar') != 'cifar':
        return malicious_train(model, dataset, args)
    if getattr(args, 'poison_type', 'southwest') != 'southwest':
        return malicious_train(model, dataset, args)

    attack_case = getattr(args, 'attack_case', 'edge-case')
    ood_train, _ = _load_southwest_ood(args, attack_case)
    if ood_train is None:
        # 资源缺失回退
        return malicious_train(model, dataset, args)

    ood_train_np, ood_train_lbl = ood_train

    model.train()
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=getattr(args, 'lr', 0.1), momentum=getattr(args, 'momentum', 0.5)
    )
    epoch_loss = []
    train_loader = DataLoader(dataset, batch_size=getattr(args, 'local_bs', 64), shuffle=True)
    for iter in range(getattr(args, 'local_ep', 1)):
        batch_loss = []
        for images, labels in train_loader:
            # 对 batch 进行 edge-case/OOD 混合；Edge-Case 已通过标签生成策略写入 ood_train_lbl
            mixed_images, mixed_labels = _sample_and_mix_edgecase(
                images, labels, args, ood_train_np, ood_train_lbl
            )
            mixed_images, mixed_labels = mixed_images.to(args.device), mixed_labels.to(args.device)
            model.zero_grad()
            log_probs = model(mixed_images)
            loss = error(log_probs, mixed_labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return model.state_dict(), (sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0)
    
def add_trigger(args, image):
        if args.trigger == 'square':
            pixel_max = torch.max(image) if torch.max(image)>1 else 1
            
            image[:,args.triggerY:args.triggerY+5,args.triggerX:args.triggerX+5] = pixel_max
        elif args.trigger == 'pattern':
            pixel_max = torch.max(image) if torch.max(image)>1 else 1
            image[:,args.triggerY+0,args.triggerX+0] = pixel_max
            image[:,args.triggerY+1,args.triggerX+1] = pixel_max
            image[:,args.triggerY-1,args.triggerX+1] = pixel_max
            image[:,args.triggerY+1,args.triggerX-1] = pixel_max
        elif args.trigger == 'watermark':
            if args.watermark is None:
                args.watermark = cv2.imread('./utils/watermark.png', cv2.IMREAD_GRAYSCALE)
                args.watermark = cv2.bitwise_not(args.watermark)
                args.watermark = cv2.resize(args.watermark, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(args.watermark)
                args.watermark = args.watermark.astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                args.watermark *= pixel_max_dataset
            max_pixel = max(np.max(args.watermark),torch.max(image))
            image = (image.cpu() + args.watermark).to(args.gpu)
            image[image>max_pixel]=max_pixel
        elif args.trigger == 'apple':
            if args.apple is None:
                args.apple = cv2.imread('./utils/apple.png', cv2.IMREAD_GRAYSCALE)
                args.apple = cv2.bitwise_not(args.apple)
                args.apple = cv2.resize(args.apple, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(args.apple)
                args.apple = args.apple.astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                args.apple *= pixel_max_dataset
            max_pixel = max(np.max(args.apple),torch.max(image))
            image += (image.cpu() + args.apple).to(args.gpu)
            image[image>max_pixel]=max_pixel
        return image
    
def test(model, dataset, args, backdoor=True):    
    if backdoor == True:
        acc_test, _, back_acc = test_img(
                            copy.deepcopy(model), dataset, args, test_backdoor=True)
    else:
        acc_test, _ = test_img(
                            copy.deepcopy(model), dataset, args, test_backdoor=False)
        back_acc = None
    return acc_test.item(), back_acc


def FLS(model_benign, model_malicious, BSR, mal_val_dataset,args):
    good_weight = model_benign.state_dict()
    bad_weight = model_malicious.state_dict()
    key_arr = []
    value_arr = []
    net3 = copy.deepcopy(model_benign)
    
    for key, var in model_benign.named_parameters():
        # if "bias" in key:
        #     continue
        param = copy.deepcopy(bad_weight)
        param[key] = var
        net3.load_state_dict(param)
        acc, _, back_acc2 = test_img(net3, mal_val_dataset, args, test_backdoor=True)
        key_arr.append(key)
        value_arr.append(back_acc2 - BSR)

    return key_arr, value_arr


def BLS(key_arr, value_arr,model_benign, model_malicious, BSR, mal_val_dataset,args, threshold=0.8):
    good_weight = model_benign.state_dict()
    bad_weight = model_malicious.state_dict()
    n = 1
    temp_BSR = 0
    attack_list = []
    np_key_arr = np.array(key_arr)
    net3= copy.deepcopy(model_benign)
    while(temp_BSR<BSR*threshold and n <=len(key_arr)):
        minValueIdx = heapq.nsmallest(n, range(len(value_arr)), value_arr.__getitem__)
        attack_list = list(np_key_arr[minValueIdx])
        param = copy.deepcopy(good_weight)
        for layer in attack_list:
            param[layer] = bad_weight[layer]
        net3.load_state_dict(param)
        acc, _, temp_BSR = test_img(net3, mal_val_dataset, args, test_backdoor=True)
        # print(attack_list)
        # print(temp_BSR)
        n += 1
    print(attack_list)
    return attack_list



def layer_analysis_no_acc(model_param,args,mal_train_dataset, mal_val_dataset,threshold=0.8):
    if args.model == 'resnet':
        model = ResNet18().to(args.device)
    elif args.model == 'VGG':
        model = vgg19_bn().to(args.device)
    elif args.model == 'rlr_mnist':
        model = get_model('fmnist').to(args.device)
    param1 = model_param
    model.load_state_dict(param1)
    
    model_benign = copy.deepcopy(model)
    acc, backdoor = test(copy.deepcopy(model_benign),mal_train_dataset,args)
    if args.dataset=='cifar':
        min_acc=93
    else:
        min_acc=90
    num_time=0
    while(acc<min_acc):
        benign_train(model_benign,mal_train_dataset,args)
        num_time += 1
        if num_time%4==0:
            acc, _ = test(copy.deepcopy(model_benign),mal_train_dataset,args, False)
            model = model_benign
            if num_time > 30:
                if acc > 80:
                    break
                else:
                    attack_list = []
                    return attack_list
        
    # benign_train(model_benign,mal_train_dataset,args)
    model_malicious = copy.deepcopy(model)
    model_malicious.load_state_dict(model.state_dict())
    malicious_train(model_malicious,mal_train_dataset,args)
    acc, back_acc = test(model_malicious,mal_val_dataset,args)
    
    # print("malicious train test", test(model_malicious,mal_train_dataset,args))
    acc, backdoor = test(model_benign,mal_val_dataset,args)
    print("benign model testset result(acc/backdoor):",acc, backdoor)
    acc, back_acc = test(model_malicious,mal_val_dataset,args)
    print("malicious model testset result(acc/backdoor):",acc, back_acc)
    
    good_weight = model_benign.state_dict()
    bad_weight = model_malicious.state_dict()
    temp_weight = copy.deepcopy(good_weight)
    for layer in args.attack_layers:
        temp_weight[layer] = bad_weight[layer]
    temp_model = copy.deepcopy(model_benign)
    temp_model.load_state_dict(temp_weight)
    acc, test_model_backdoor = test(temp_model,mal_val_dataset,args)
    if test_model_backdoor > threshold*back_acc:
        print(test_model_backdoor,">",threshold*back_acc,"SKIP")
        return args.attack_layers
    
    key_arr, value_arr = FLS(model_benign, model_malicious, back_acc, mal_val_dataset,args)
    threshold = args.tau
    attack_list = BLS(key_arr, value_arr,model_benign, model_malicious, back_acc, mal_val_dataset,args,threshold=threshold)
    print("finish identification")
    return attack_list





def get_attacker_dataset(args):
    
    if args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            client_proportion = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
        else:
            client_proportion = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
    elif args.dataset == "fashion_mnist":
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        dataset_train = datasets.FashionMNIST(
            '../data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(
            '../data/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            client_proportion = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
        else:
            client_proportion = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
            
    data_list=[]
    begin_pos=0
    malicious_client_num = int(args.num_users*args.malicious)
    for i in range(begin_pos, begin_pos+malicious_client_num):
        data_list.extend(client_proportion[i])
    attacker_label = []
    for i in range(len(data_list)):
        attacker_label.append(dataset_train.targets[data_list[i]])
    attacker_label = np.array(attacker_label)
    client_dataset = []
    for i in range(len(data_list)):
        client_dataset.append(dataset_train[data_list[i]])
    mal_train_dataset, mal_val_dataset = split_dataset(client_dataset)
    return mal_train_dataset, mal_val_dataset  

def split_dataset(dataset):
    num_dataset = len(dataset)
    # random
    data_distribute = np.random.permutation(num_dataset)
    malicious_dataset=[]
    mal_val_dataset=[]
    mal_train_dataset=[]
    for i in range(num_dataset):
        malicious_dataset.append(dataset[data_distribute[i]])
        if i < num_dataset//4:
            mal_val_dataset.append(dataset[data_distribute[i]])
        else:
            mal_train_dataset.append(dataset[data_distribute[i]])
    return mal_train_dataset, mal_val_dataset
        

def get_attack_layers_no_acc(model_param,args):
    mal_train_dataset, mal_val_dataset = get_attacker_dataset(args)
    return layer_analysis_no_acc(model_param,args,mal_train_dataset, mal_val_dataset)


def pgd_parameter_attack(model, batch_iterable, args,
                         eps=None, proj='l2', project_frequency=None, adv_lr=None):
    """
    使用参数空间 PGD 对模型进行一轮（或多轮）对抗更新，并进行范数球投影。

    参数:
    - model: 需攻击的模型（已在正确设备上）
    - batch_iterable: 任意可迭代的 (images, labels) 批次集合（如 DataLoader 或生成器）
                      建议传入已注入触发器/投毒的数据批次
    - args: 全局配置对象，使用其中的 `lr`, `momentum`, `device`, `local_ep` 等
    - eps: 投影半径，默认为 `args.pgd_eps`
    - proj: 投影范数类型，'l2' 或 'linf'，默认为 `args.pgd_proj`
    - project_frequency: 投影频率，默认为 `args.pgd_project_frequency`
    - adv_lr: 对抗更新的学习率，默认为 `args.pgd_adv_lr`

    返回:
    - state_dict, avg_loss
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if adv_lr is None:
        adv_lr = getattr(args, 'pgd_adv_lr', 1e-3)
    adv_optimizer = torch.optim.SGD(model.parameters(), lr=adv_lr, momentum=args.momentum)

    if eps is None:
        eps = getattr(args, 'pgd_eps', 5e-4)
    if project_frequency is None:
        project_frequency = getattr(args, 'pgd_project_frequency', 1)
    if proj is None:
        proj = getattr(args, 'pgd_proj', 'l2')

    # 保存初始参数向量
    original_params = [p.detach().clone() for p in model.parameters()]
    original_vec = parameters_to_vector(original_params)

    epoch_losses = []
    for epoch in range(getattr(args, 'local_ep', 1)):
        batch_losses = []
        for batch_idx, (images, labels) in enumerate(batch_iterable):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            adv_optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()

            if proj == 'linf':
                # 手动梯度步 + L_inf 投影到原参数的 +-eps 矩形域
                step_size = adv_lr
                params_list = list(model.parameters())
                for i in range(len(params_list)):
                    params_list[i].data = params_list[i].data - step_size * params_list[i].grad.data
                    diff = params_list[i] - original_params[i]
                    params_list[i].data = torch.max(torch.min(diff, torch.tensor(eps, device=diff.device)),
                                                    torch.tensor(-eps, device=diff.device)) + original_params[i]
            else:
                # 标准对抗更新 + L2 范数球投影
                adv_optimizer.step()
                # 仅按频率进行中间投影，最终在epoch末进行一次强制投影
                if project_frequency and project_frequency > 0:
                    if batch_idx % project_frequency == 0:
                        current_vec = parameters_to_vector(list(model.parameters()))
                        diff_vec = current_vec - original_vec
                        diff_norm = torch.norm(diff_vec)
                        if diff_norm > eps:
                            proj_vec = eps * diff_vec / diff_norm + original_vec
                            vector_to_parameters(proj_vec, list(model.parameters()))

            batch_losses.append(loss.item())
        # epoch 末进行一次强制 L2 投影，避免中途频繁投影过度抑制更新
        if proj == 'l2':
            current_vec = parameters_to_vector(list(model.parameters()))
            diff_vec = current_vec - original_vec
            diff_norm = torch.norm(diff_vec)
            if diff_norm > eps:
                proj_vec = eps * diff_vec / diff_norm + original_vec
                vector_to_parameters(proj_vec, list(model.parameters()))
        # 轻量日志：记录范数以便调参
        try:
            print(f"[PGD] epoch {epoch}: ||Δθ||={diff_norm.item():.6f}, eps={eps}, adv_lr={adv_lr}, freq={project_frequency}, proj={proj}")
        except Exception:
            pass
        if batch_losses:
            epoch_losses.append(sum(batch_losses) / len(batch_losses))

    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    return model.state_dict(), avg_loss
