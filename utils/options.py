#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # save file 
    parser.add_argument('--save', type=str, default='save',
                        help="dic to save results (ending without /)")
    parser.add_argument('--init', type=str, default='None',
                        help="location of init model")
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help="rounds of training")
    parser.add_argument('--num_users', type=int,
                        default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help="the fraction of clients: C")
    parser.add_argument('--malicious',type=float,default=0.2, help="proportion of mailicious clients")
    
    #***** badnet labelflip layerattack updateflip get_weight layerattack_rev layerattack_ER****
    parser.add_argument('--attack', type=str,
                        default='badnet', help='attack method')

    # Edge-Case/OOD 攻击配置
    parser.add_argument('--attack_case', type=str, default='edge-case',
                        help='edge-case mode: edge-case | normal-case | almost-edge-case')
    parser.add_argument('--poison_type', type=str, default='southwest',
                        help='poison/OOD type, e.g., southwest')
    parser.add_argument('--edge_target_label', type=int, default=9,
                        help='edge-case 目标标签（cifar默认 truck=9），用于语义/OOD攻击评估与训练')
    parser.add_argument('--edge_poison_n', type=int, default=32,
                        help='每个批次混入的 OOD 样本数（控制 edge-case 强度）')
    
    parser.add_argument('--poison_frac', type=float, default=0.3, 
                        help="fraction of dataset to corrupt for backdoor attack, 1.0 for layer attack")

    # *****local_ep = 3, local_bs=50, lr=0.1*******
    parser.add_argument('--local_ep', type=int, default=3,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: B")

    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--mul',type=float,default=2.0,help="multiple of the length limit")

    # model arguments
    #*************************model******************************#
    # resnet cnn VGG mlp Mnist_2NN Mnist_CNN resnet20 rlr_mnist
    parser.add_argument('--model', type=str,
                        default='Mnist_CNN', help='model name')

    # other arguments
    #*************************dataset*******************************#
    # fashion_mnist mnist cifar
    parser.add_argument('--dataset', type=str,
                        default='mnist', help="name of dataset")
    
    
    

    parser.add_argument('--defence', type=str,
                        default='avg', help="strategy of defence")
    # parser.add_argument('--iid', action='store_true',
    #                     help='whether i.i.d or not')
    parser.add_argument('--iid', type=int, default=1,
                        help='whether i.i.d or not')

 #************************atttack_label********************************#
    parser.add_argument('--attack_label', type=int, default=5,
                        help="trigger for which label")
    # attack_goal=-1 is all to one
    parser.add_argument('--attack_goal', type=int, default=7,
                        help="trigger to which label")
    # --attack_begin 70 means accuracy is up to 70 then attack
    parser.add_argument('--attack_begin', type=int, default=0,
                        help="the accuracy begin to attack")
    
    parser.add_argument('--gpu', type=int, default=1,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--robustLR_threshold', type=int, default=4, 
                        help="break ties when votes sum to 0")
    
    parser.add_argument('--server_dataset', type=int,default=0,help="number of dataset in server")
    
    parser.add_argument('--server_lr', type=float,default=1,help="number of dataset in server using in fltrust")
    
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="SGD momentum (default: 0.5)")
    
    
    parser.add_argument('--split', type=str, default='user',
                        help="train-test split type, user or sample")   
    #*********trigger info*********
    #  square  apple  watermark  
    parser.add_argument('--trigger', type=str, default='square',
                        help="Kind of trigger")  
    # mnist 28*28  cifar10 32*32
    parser.add_argument('--triggerX', type=int, default='0',
                        help="position of trigger x-aix") 
    parser.add_argument('--triggerY', type=int, default='0',
                        help="position of trigger y-aix")
    
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--wrong_mal', type=int, default=0)
    parser.add_argument('--right_ben', type=int, default=0)
    
    parser.add_argument('--mal_score', type=float, default=0)
    parser.add_argument('--ben_score', type=float, default=0)
    
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0.001)
    parser.add_argument('--all_clients', action='store_true',
                        help='aggregation over all clients')
    
    # weak-dp 防御相关参数
    parser.add_argument('--weakdp_norm_bound', type=float, default=1.0,
                        help='norm bound for weak-dp defense (default: 1.0)')
    parser.add_argument('--weakdp_noise_stddev', type=float, default=0.01,
                        help='noise standard deviation for weak-dp defense (default: 0.01)') 

    # PGD 参数空间攻击相关超参数
    parser.add_argument('--pgd_eps', type=float, default=5e-4,
                        help='PGD 范数球半径（参数空间）')
    parser.add_argument('--pgd_proj', type=str, default='l2',
                        help='PGD 投影范数类型: l2 或 linf')
    parser.add_argument('--pgd_adv_lr', type=float, default=1e-3,
                        help='PGD 对抗更新学习率（参数空间）')
    parser.add_argument('--pgd_project_frequency', type=int, default=1,
                        help='PGD 投影频率（按batch计），l2模式通常每batch都投影')

    # 评估阶段触发器图片保存开关（默认关闭，避免频繁写盘导致阻塞）
    parser.add_argument('--save_test_trigger', type=int, default=0,
                        help='是否在评估时保存触发器示例图片(0/1)，默认0不保存')


    # Model Replacement Attack scaling factor
    parser.add_argument('--scale', type=float, default=1.0,
                        help='scaling factor for model replacement attack (default: 1.0, no scaling)')

    args = parser.parse_args()
    return args
