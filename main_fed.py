#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# 说明：
# 本脚本实现联邦学习训练主流程（含后门攻击与多种防御），
# 包括数据集加载与划分、全局模型构建、每轮客户端选择与本地训练、
# 聚合与防御策略应用、指标记录与可视化，以及最终测试与日志保存。
# 重要变量：
# - `w_glob`：全局模型参数（state_dict）
# - `w_locals`：各参与客户端训练后的本地模型参数集合
# - `w_updates`：各客户端相对全局模型的参数更新量（通过 `get_update` 计算）
# - `args`：命令行解析得到的配置对象，控制模型、数据、攻击与防御等
# 运行输出会保存到 `./logs/...` 和 `./{args.save}/...` 目录中。

from random import random
from models.test import test_img, test_edgecase_asr, test_edgecase_asr_ood
from models.Fed import FedAvg
from models.Nets import ResNet18, vgg19_bn, vgg19, get_model, vgg16_w05_bn

from models.MaliciousUpdate import LocalMaliciousUpdate
from models.Update import LocalUpdate
from utils.info import print_exp_details, write_info_to_accfile, get_base_info
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.defense import fltrust, multi_krum, get_update, RLR, flame, newFlame, fools_gold, weak_dp
from utils.scram import scram,scram_without_privacy

import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
import random
import time
import sys
from datetime import datetime
import logging
import csv
import math
from torch.utils.tensorboard import SummaryWriter
import pickle

matplotlib.use('Agg')




def write_file(filename, accu_list, back_list, args, analyse = False):
    """
    将当前实验的主任务准确率与后门准确率写入到指定文件。
    参数：
    - filename: 输出文件路径。
    - accu_list: 主任务准确率列表（每轮累积）。
    - back_list: 后门准确率列表（每轮累积）。
    - args: 命令行参数对象，可能用于保存额外信息（如krum距离）。
    - analyse: 若为 True，则在文件末尾追加统计信息（BBSR、ABSR、最佳主任务acc），并返回三者。
    返回：
    - 当 analyse=True 时，返回 (best_acc, average_back, best_back)，否则无返回。
    """
    write_info_to_accfile(filename, args)
    f = open(filename, "a")
    f.write("main_task_accuracy=")
    f.write(str(accu_list))
    f.write('\n')
    f.write("backdoor_accuracy=")
    f.write(str(back_list))
    if args.defence == "krum":
        krum_file = filename+"_krum_dis"
        torch.save(args.krum_distance,krum_file)
    if analyse == True:
        need_length = len(accu_list)//10
        acc = accu_list[-need_length:]
        back = back_list[-need_length:]
        best_acc = round(max(acc),2)
        average_back=round(np.mean(back),2)
        best_back=round(max(back),2)
        f.write('\n')
        f.write('BBSR:')
        f.write(str(best_back))
        f.write('\n')
        f.write('ABSR:')
        f.write(str(average_back))
        f.write('\n')
        f.write('max acc:')
        f.write(str(best_acc))
        f.write('\n')
        f.close()
        return best_acc, average_back, best_back
    f.close()


def central_dataset_iid(dataset, dataset_size):
    """
    从完整测试集均匀随机采样 `dataset_size` 个样本索引，作为服务器的可信中心数据。
    该数据用于如 `fltrust` 等防御方法中的服务器端参考。
    """
    all_idxs = [i for i in range(len(dataset))]
    central_dataset = set(np.random.choice(
        all_idxs, dataset_size, replace=False))
    return central_dataset

def test_mkdir(path):
    """
    如果目录 `path` 不存在则创建，用于初始化日志与输出目录。
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def append_flame_epoch(save_path, epoch, w_locals, w_updates, w_glob, args):
    """
    将同一个训练过程中的每个 epoch 的 flame/newFlame 输入参数
    (epoch, w_locals, w_updates, w_glob, args) 顺序追加到同一个文件中。

    - 使用 pickle 流以 'ab' 方式追加，避免一次性载入全部数据造成显存/内存压力。
    - 自动将张量移动到 CPU 并 detach，减少 GPU 显存占用并避免跨设备反序列化问题。

    参数:
    - save_path: 追加保存的目标文件路径，例如 `os.path.join(log_dir, 'flame_io_stream.pkl')`
    - epoch: 当前轮次编号（int）
    - w_locals, w_updates: 列表[dict[str, Tensor]]，每个客户端的权重或更新
    - w_glob: dict[str, Tensor]，当前全局模型权重
    - args: argparse.Namespace，当前配置对象
    """
    def _to_cpu_tensor(x):
        return x.detach().cpu() if torch.is_tensor(x) else x

    def _dict_to_cpu(d):
        return {k: _to_cpu_tensor(v) for k, v in d.items()}

    def _list_of_dicts_to_cpu(lst):
        return [_dict_to_cpu(sd) for sd in lst]

    try:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        payload = {
            'epoch': epoch,
            'w_locals': _list_of_dicts_to_cpu(w_locals),
            'w_updates': _list_of_dicts_to_cpu(w_updates),
            'w_glob': _dict_to_cpu(w_glob),
            'args': args,
        }
        with open(save_path, 'ab') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[append_flame_epoch] 轮次 {epoch} 已追加到: {save_path}")
    except Exception as e:
        print(f"[append_flame_epoch] 追加失败: {e}")


def iter_flame_epochs(load_path):
    """
    顺序读取由 `append_flame_epoch` 追加保存的文件内容，逐个返回每个 epoch 的数据。

    使用示例：
    for epoch, w_locals, w_updates, w_glob, args in iter_flame_epochs(path):
        # 在这里使用该 epoch 的数据进行分析/复现/评估，然后继续读取下一条

    注意：该函数不会一次性将所有数据载入内存，而是流式逐条读取，避免显存不足。
    """
    try:
        with open(load_path, 'rb') as f:
            while True:
                try:
                    payload = pickle.load(f)
                except EOFError:
                    break
                except Exception as e:
                    print(f"[iter_flame_epochs] 读取失败: {e}")
                    break

                yield (
                    payload.get('epoch'),
                    payload.get('w_locals'),
                    payload.get('w_updates'),
                    payload.get('w_glob'),
                    payload.get('args'),
                )
    except FileNotFoundError:
        print(f"[iter_flame_epochs] 文件未找到: {load_path}")
    except Exception as e:
        print(f"[iter_flame_epochs] 打开文件失败: {e}")


if __name__ == '__main__':
    

    # parse args
    args = args_parser()
    
    # 在训练开始之前初始化SummaryWriter
    current_time = datetime.now()
    # 格式化时间字符串，例如：2023-04-01_15-30-45
    current_time_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    
    # 创建日志目录
    base_info = get_base_info(args)
    log_dir = './logs/{}/{}/{}/{}/{}'.format(args.save, args.attack, args.defence, args.mul, base_info)
    test_mkdir('./logs')
    test_mkdir('./logs/{}'.format(args.save))
    test_mkdir('./logs/{}/{}'.format(args.save, args.attack))
    test_mkdir('./logs/{}/{}/{}'.format(args.save, args.attack, args.defence))
    test_mkdir('./logs/{}/{}/{}/{}'.format(args.save, args.attack, args.defence, args.mul))
    test_mkdir(log_dir)
    
    # 初始化TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # 设置日志文件
    log_file_path = '{}/logFile.log'.format(log_dir)
    logFile = open(log_file_path, 'a')
    print(f'TensorBoard logs will be saved to: {log_dir}')
    print(f'Log file will be saved to: {log_file_path}')
    
    # 初始化每个 epoch 计时结果的 CSV 文件（位于同一日志目录）
    epoch_times_csv = os.path.join(log_dir, 'epoch_times.csv')
    if not os.path.isfile(epoch_times_csv):
        try:
            with open(epoch_times_csv, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["epoch", "seconds"])
        except Exception as e:
            print(f"Warning: Could not initialize epoch times CSV: {e}")
    
    # 保存原始的标准输出对象
    stdout = sys.stdout
    
    # 将输出重定向到文件
    sys.stdout = logFile
    
    print(f"Experiment started at: {current_time_str}")
    print(f"Arguments: {args}")
    print(f"Log directory: {log_dir}")
    
    # 恢复原始的标准输出对象以便后续正常输出
    sys.stdout = stdout

    
    
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # 根据是否可用的GPU与命令行参数选择训练设备（CUDA或CPU）
    test_mkdir('./'+args.save)
    print_exp_details(args)
    
    # load dataset and split users
    # 加载指定数据集，并根据 `args.iid` 或预生成的映射将数据划分到各个客户端（dict_users）。
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion_mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        dataset_train = datasets.FashionMNIST(
            '../data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(
            '../data/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
        else:
            dict_users = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
        else:
            dict_users = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    # 按照参数选择模型结构（VGG、ResNet、或自定义fmnist模型），并迁移至指定设备。
    if args.model == 'VGG' and args.dataset == 'cifar':
        # net_glob = vgg19_bn().to(args.device)
        # Default to a lighter VGG for CIFAR to save memory
        net_glob = vgg16_w05_bn().to(args.device)
    elif args.model == "resnet" and args.dataset == 'cifar':
        net_glob = ResNet18().to(args.device)
    elif args.model == "rlr_mnist" or args.model == "cnn":
        net_glob = get_model('fmnist').to(args.device)
    else:
        exit('Error: unrecognized model')
    
    net_glob.train()

    # copy weights
    # 初始化全局权重字典，用于后续聚合和更新。
    w_glob = net_glob.state_dict()
    
    # 添加模型结构图到TensorBoard
    try:
        # 创建一个示例输入来生成模型图
        if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
            sample_input = torch.randn(1, 1, 28, 28).to(args.device)
        elif args.dataset == 'cifar':
            sample_input = torch.randn(1, 3, 32, 32).to(args.device)
        else:
            sample_input = torch.randn(1, *img_size).to(args.device)
        
        # 添加模型图到TensorBoard
        writer.add_graph(net_glob, sample_input)
        print("Model graph added to TensorBoard successfully")
    except Exception as e:
        print(f"Warning: Could not add model graph to TensorBoard: {e}")

    # training
    # 训练过程中的度量初始化：记录每轮训练损失、验证准确率/损失等。
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    
    if math.isclose(args.malicious, 0):
        backdoor_begin_acc = 100    # 如果恶意客户端个数为0，永远都不攻击
    else:
        backdoor_begin_acc = args.attack_begin  # 当主任务准确率超过该阈值后才开始触发攻击
    central_dataset = central_dataset_iid(dataset_test, args.server_dataset)
    # 为服务器端创建一个中心（可信）数据集索引，用于某些防御（例如 fltrust）
    filename = './'+args.save+'/accuracy_file_{}.txt'.format(base_info)
    # 指定当前实验的准确率记录文件
    
    if args.init != 'None':
        param = torch.load(args.init)
        net_glob.load_state_dict(param)
        print("load init model")
        # 若指定初始化模型权重文件，则加载为全局模型初始参数

        
    val_acc_list, net_list = [0], []
    backdoor_acculist = [0]

    args.attack_layers=[]
    # 记录每轮选择的恶意客户端编号（列表的列表，每轮一个子列表）
    args.attackers_chosen = []
    # 记录按层攻击时的层选择历史（部分攻击方法会更新该列表）
    
    if args.attack == "dba":
        args.dba_sign=0
        # DBA（分布式后门）攻击的轮次计数，用于轮流选择不同触发器
    if args.defence == "krum":
        args.krum_distance=[]
        # KRUM防御过程中的距离统计，将在训练结束后保存
        
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        # 若采用全量聚合，则预先为每个客户端填充一份全局参数副本
    for iter in range(args.epochs):
        loss_locals = []
        # 本轮开始计时
        epoch_start = time.time()
        if not args.all_clients:
            w_locals = []
            w_updates = []
        # `w_locals` 用于收集本轮参与客户端的本地权重；`w_updates` 收集相对全局的更新量
        m = max(int(args.frac * args.num_users), 1)
        # 随机选取本轮参与训练的客户端集合，数量为 frac*num_users（至少为1）
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        if val_acc_list[-1] > backdoor_begin_acc:
            attack_number = int(args.malicious * m)
        else:
            attack_number = 0
        # 当主任务准确率超过阈值后，按照恶意比例在本轮参与客户端中分配攻击者数量
        
        # 本轮记录的恶意客户端编号
        round_attackers = []
        
        # --- Pre-select unique malicious indices for this round ---
        current_round_malicious_indices = []
        if attack_number > 0:
            total_malicious_pool_size = int(args.num_users * args.malicious)
            # Ensure we don't try to select more than available
            num_to_select = min(attack_number, total_malicious_pool_size)
            if num_to_select > 0:
                # Use random.sample to get unique indices
                current_round_malicious_indices = random.sample(range(total_malicious_pool_size), num_to_select)
        mal_ptr = 0
        # ----------------------------------------------------------

        for num_turn, idx in enumerate(idxs_users):
            if attack_number > 0:
                attack = True
            else:
                attack = False
            if attack == True:
                # Use pre-selected unique index
                if mal_ptr < len(current_round_malicious_indices):
                    idx = current_round_malicious_indices[mal_ptr]
                    mal_ptr += 1
                else:
                    # Fallback (should not happen if logic is correct)
                    idx = random.randint(0, int(args.num_users * args.malicious))

                # if args.attack == "dba":
                #     num_dba_attacker = int(args.num_users * args.malicious)
                #     dba_group = num_dba_attacker//4
                #     idx = args.dba_sign % (4*dba_group)
                #     args.dba_sign+=1
                
                # Keep dba_sign increment if needed for other logic, but don't overwrite idx
                if args.attack == "dba":
                    args.dba_sign += 1

                local = LocalMaliciousUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], order=idx)

                if args.attack == "layerattack_ER_his" or args.attack == "LFA" or args.attack == "LPA":
                    w, loss, args.attack_layers = local.train(
                        net=copy.deepcopy(net_glob).to(args.device), test_img = test_img)
                else:
                    w, loss = local.train(
                        net=copy.deepcopy(net_glob).to(args.device), test_img = test_img)
                print("client", idx, "--attack--")
                # 记录该轮的恶意客户端编号
                try:
                    round_attackers.append(int(idx))
                except Exception:
                    round_attackers.append(idx)
                attack_number -= 1
            else:
                # 正常客户端执行标准本地训练流程
                local = LocalUpdate(
                    args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(
                    net=copy.deepcopy(net_glob).to(args.device))
            # 记录该客户端相对全局模型的参数更新（w - w_glob）
            w_updates.append(get_update(w, w_glob))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # 将本轮的恶意客户端记录追加到 args
        try:
            args.attackers_chosen.append(round_attackers)
        except Exception:
            pass

        if args.defence == 'avg':  # no defence
            w_glob = FedAvg(w_locals)
            # append_flame_epoch(os.path.join(log_dir, 'flame_io_stream.pkl'), iter, w_locals, w_updates, w_glob, args)
            # 标准平均聚合（无防御）
        elif args.defence == 'krum':  # single krum
            selected_client = multi_krum(w_updates, 1, args)
            # print(args.krum_distance)
            w_glob = w_locals[selected_client[0]]
            # w_glob = FedAvg([w_locals[i] for i in selected_clinet])
            # KRUM防御：根据更新距离选择一个可信客户端的权重作为全局
        elif args.defence == 'RLR':
            w_glob = RLR(copy.deepcopy(net_glob), w_updates, args)
            # RLR防御：鲁棒线性回归式聚合（详见utils.defense.RLR）
        elif args.defence == 'fltrust':
            local = LocalUpdate(
                args=args, dataset=dataset_test, idxs=central_dataset)
            fltrust_norm, loss = local.train(
                net=copy.deepcopy(net_glob).to(args.device))
            fltrust_norm = get_update(fltrust_norm, w_glob)
            w_glob = fltrust(w_updates, fltrust_norm, w_glob, args)
        elif args.defence in ['foolsgold', 'fg']:
            # 使用FoolsGold基于更新向量的相似性进行加权聚合
            w_glob = fools_gold(w_updates, copy.deepcopy(w_glob), args)
            # FLTrust防御：使用中心数据的更新作为参考，对各客户端更新进行加权归一化
        elif args.defence == 'flame':
            # append_flame_epoch(os.path.join(log_dir, 'flame_io_stream.pkl'), iter, w_locals, w_updates, w_glob, args)
            w_glob = flame(w_locals,w_updates,w_glob, args)
            # FLAME防御：过滤与聚合策略结合（详见utils.defense.flame）
            
        elif args.defence == 'newFlame':
            # append_flame_epoch(os.path.join(log_dir, 'flame_io_stream.pkl'), iter, w_locals, w_updates, w_glob, args)
            w_glob = newFlame(w_locals,w_updates,w_glob, args)
            # newFlame：改进版FLAME，加入内存监控与更强的鲁棒策略（详见utils.defense.newFlame）
        elif args.defence.lower() in ['foolsgold', 'fg']:
            # FoolsGold：基于客户端更新相似性的加权聚合
            w_glob = fools_gold(w_updates, w_glob, args)
        elif args.defence == 'scram':
            # append_flame_epoch(os.path.join(log_dir, 'flame_io_stream.pkl'), iter, w_locals, w_updates, w_glob, args)
            w_glob = scram(w_locals,w_updates,w_glob, args)
        elif args.defence == 'scram_without_privacy':
            # append_flame_epoch(os.path.join(log_dir, 'flame_io_stream.pkl'), iter, w_locals, w_updates, w_glob, args)
            w_glob = scram_without_privacy(w_locals,w_updates,w_glob, args)
        elif args.defence == 'weak-dp':
            # weak-dp防御：简化的差分隐私防御，结合范数裁剪和噪声添加
            from utils.defense import weak_dp
            w_glob = weak_dp(w_locals, w_updates, w_glob, args)
        else:
            print("Wrong Defense Method")
            os._exit(0)
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # 将聚合后的参数写回到全局模型

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        # 本轮客户端训练损失的平均值，用于监控训练过程
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # 在每轮训练结束后记录损失和准确率到TensorBoard
        writer.add_scalar('Loss/train', loss_avg, iter)
        
        # 记录学习率（如果有的话）
        if hasattr(args, 'lr'):
            writer.add_scalar('Learning_Rate', args.lr, iter)
        
        # 记录参与训练的客户端数量
        writer.add_scalar('Clients/participating', len(idxs_users), iter)
        writer.add_scalar('Clients/malicious', attack_number if 'attack_number' in locals() else 0, iter)
        
        # 记录防御机制信息
        writer.add_text('Defense_Method', args.defence, iter)
        


        if iter % 1 == 0:
            # 常规像素触发器后门评估
            acc_test, loss_test_val, back_acc = test_img(
                net_glob, dataset_test, args, test_backdoor=True)
            print("Main accuracy: {:.2f}".format(acc_test))
            print("Backdoor accuracy: {:.2f}".format(back_acc))
            val_acc_list.append(acc_test.item())

            # Edge-Case/OOD 评估（仅在 CIFAR + southwest 时启用）
            edgecase_enabled = (args.dataset == 'cifar' and getattr(args, 'poison_type', 'southwest') == 'southwest')
            edge_asr = None
            if edgecase_enabled:
                try:
                    edge_asr = test_edgecase_asr(net_glob, dataset_test, args)
                    print("Edge-Case ASR: {:.2f}".format(edge_asr))
                    # OOD 测试集直接评估 ASR（若资源存在）
                    edge_asr_ood = test_edgecase_asr_ood(net_glob, args)
                    if edge_asr_ood is not None:
                        print("Edge-Case ASR (OOD): {:.2f}".format(edge_asr_ood))
                        writer.add_scalar('Accuracy/edgecase_asr_ood', edge_asr_ood, iter)
                except Exception as e:
                    print(f"Edge-Case ASR evaluation failed: {e}")

            backdoor_acculist.append(back_acc)
            write_file(filename, val_acc_list, backdoor_acculist, args)
        
            # 记录测试准确率和后门准确率到TensorBoard
            writer.add_scalar('Accuracy/test', acc_test, iter)
            writer.add_scalar('Accuracy/backdoor', back_acc, iter)
            if edge_asr is not None:
                writer.add_scalar('Accuracy/edgecase_asr', edge_asr, iter)
            writer.add_scalar('Loss/test', loss_test_val if loss_test_val is not None else 0, iter)
            
            # 记录准确率差异（主任务准确率 - 后门准确率）
            acc_diff = acc_test.item() - back_acc
            writer.add_scalar('Accuracy/difference', acc_diff, iter)
            
            # 记录累计最佳准确率
            best_acc_so_far = max(val_acc_list)
            writer.add_scalar('Accuracy/best_so_far', best_acc_so_far, iter)
        
        # 统计并记录本轮（epoch）总耗时
        epoch_seconds = time.time() - epoch_start
        writer.add_scalar('Time/epoch_seconds', epoch_seconds, iter)
        try:
            with open(epoch_times_csv, 'a', newline='') as f:
                csv.writer(f).writerow([iter, round(epoch_seconds, 6)])
        except Exception as e:
            print(f"Warning: Could not append epoch time to CSV: {e}")
        print(f"Epoch {iter} runtime: {epoch_seconds:.3f} s")
    
    best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, args, True)
    
    # 记录最终结果到TensorBoard
    writer.add_scalar('Final/best_accuracy', best_acc, args.epochs)
    writer.add_scalar('Final/average_backdoor_success_rate', absr, args.epochs)
    writer.add_scalar('Final/best_backdoor_success_rate', bbsr, args.epochs)
    
    # 添加超参数和最终结果的记录
    hparams = {
        'dataset': args.dataset,
        'model': args.model,
        'epochs': args.epochs,
        'num_users': args.num_users,
        'frac': args.frac,
        'local_ep': args.local_ep,
        'local_bs': args.local_bs,
        'lr': args.lr if hasattr(args, 'lr') else 0.01,
        'malicious': args.malicious,
        'defence': args.defence,
        'attack': args.attack if hasattr(args, 'attack') else 'none',
        'iid': args.iid
    }
    
    metrics = {
        'best_accuracy': best_acc,
        'average_backdoor_success_rate': absr,
        'best_backdoor_success_rate': bbsr
    }
    
    writer.add_hparams(hparams, metrics)
    
    # plot loss curve
    plt.figure()
    plt.xlabel('communication')
    plt.ylabel('accu_rate')
    plt.plot(val_acc_list, label = 'main task(acc:'+str(best_acc)+'%)')
    plt.plot(backdoor_acculist, label = 'backdoor task(BBSR:'+str(bbsr)+'%, ABSR:'+str(absr)+'%)')
    plt.legend()
    title = base_info
    # plt.title(title, y=-0.3)
    plt.title(title)
    plt.savefig('./'+args.save +'/'+ title + '.pdf', format = 'pdf',bbox_inches='tight')
    
    
    # testing
    net_glob.eval()
    acc_train, loss_train_final = test_img(net_glob, dataset_train, args)
    acc_test, loss_test_final = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    
    # 记录最终训练和测试结果
    writer.add_scalar('Final/train_accuracy', acc_train, args.epochs)
    writer.add_scalar('Final/test_accuracy', acc_test, args.epochs)
    writer.add_scalar('Final/train_loss', loss_train_final if loss_train_final is not None else 0, args.epochs)
    writer.add_scalar('Final/test_loss', loss_test_final if loss_test_final is not None else 0, args.epochs)

    # 关闭TensorBoard writer
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    print("You can view the logs by running: tensorboard --logdir={}".format(log_dir))
    
    # 分析newFlame时间统计数据
    if args.defence == 'newFlame':
        try:
            from utils.timing_analysis import NewFlameTimingAnalyzer
            
            print("\n" + "="*80)
            print("newFlame 时间统计分析报告")
            print("="*80)
            
            # 创建分析器实例
            analyzer = NewFlameTimingAnalyzer(log_dir)
            
            # 生成并打印统计报告
            report = analyzer.generate_summary_report()
            print(report)
            
            # 保存详细统计结果到文件
            stats_file = os.path.join(log_dir, 'newflame_timing_statistics.txt')
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n详细统计报告已保存到: {stats_file}")
            
        except Exception as e:
            print(f"Warning: 无法生成newFlame时间统计分析报告: {e}")
            print("请确保存在newFlame时间统计数据文件")
    
    # 恢复原始的标准输出对象并关闭日志文件
    sys.stdout = logFile
    print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Final results - Best Accuracy: {best_acc}%, ABSR: {absr}%, BBSR: {bbsr}%")
    sys.stdout = stdout
    logFile.close()
    
    print(f'Log file saved to: {log_file_path}')
    print(f'TensorBoard logs saved to: {log_dir}')
    print("You can view the TensorBoard logs by running: tensorboard --logdir={}".format(log_dir))
   


