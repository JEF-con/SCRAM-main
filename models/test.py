#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import io
import cv2
from skimage import img_as_ubyte
import numpy as np
def test_img(net_g, datatest, args, test_backdoor=False):
    args.watermark = None
    args.apple = None
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    back_correct = 0
    back_num = 0
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        if test_backdoor:
            del_arr = []
            for k, image in enumerate(data):
                if test_or_not(args, target[k]):  # one2one need test
                    # data[k][:, 0:5, 0:5] = torch.max(data[k])
                    data[k] = add_trigger(args,data[k])
                    # 仅在用户开启时保存触发器示例图片，且每轮只保存一次，避免频繁写盘阻塞
                    try:
                        if getattr(args, 'save_test_trigger', 0):
                            # 通过一个简单标志避免同一轮内重复保存
                            if not hasattr(args, '_test_trigger_saved') or not args._test_trigger_saved:
                                save_img(data[k])
                                args._test_trigger_saved = True
                    except Exception:
                        pass
                    target[k] = args.attack_label
                    back_num += 1
                else:
                    target[k] = -1
            log_probs = net_g(data)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            back_correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    if test_backdoor:
        back_accu = 100.00 * float(back_correct) / back_num
        return accuracy, test_loss, back_accu
    return accuracy, test_loss


def test_edgecase_asr(net_g, datatest, args):
    """评估 Edge-Case/OOD 攻击成功率（ASR）。

    逻辑：
    - 若 `args.dataset != 'cifar'` 或 `args.poison_type != 'southwest'`，直接返回常规后门测试结果。
    - 对满足条件的样本（依据 `test_or_not`），不注入像素触发器，而是将目标标签设置为 edge-case 模式对应的标签
      （edge-case -> 9; normal/almost-edge -> 0），测量模型在该分布上的目标命中率。
    提示：datatest 仍取用 CIFAR10 测试集；该评估更贴近语义/OOD攻击的目标标签偏置。
    """
    net_g.eval()
    data_loader = DataLoader(datatest, batch_size=args.bs)
    back_correct = 0
    back_num = 0
    # 目标类：southwest 默认 truck=9；normal/almost-edge 则 airplane=0
    attack_case = getattr(args, 'attack_case', 'edge-case')
    target_label = int(getattr(args, 'edge_target_label', 9)) if attack_case == 'edge-case' else 0
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        # 构造目标标签视图：符合 test_or_not 的样本记为目标标签，其余记为 -1
        for k, _ in enumerate(data):
            if test_or_not(args, target[k]):
                target[k] = target_label
                back_num += 1
            else:
                target[k] = -1
        log_probs = net_g(data)
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        back_correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    back_accu = 100.00 * float(back_correct) / max(back_num, 1)
    return back_accu


def test_edgecase_asr_ood(net_g, args):
    """直接用 Southwest OOD 测试集评估 Edge-Case/OOD ASR。

    若资源缺失，返回 None。
    标签策略与训练一致：edge-case 使用 `args.edge_target_label`，normal/almost 使用 0。
    """
    # 仅在 CIFAR + southwest 时启用
    if getattr(args, 'dataset', 'cifar') != 'cifar' or getattr(args, 'poison_type', 'southwest') != 'southwest':
        return None
    # 读取 OOD 测试集
    try:
        from models.Attacker import _load_southwest_ood
        (_, _), (ood_test_np, ood_test_lbl) = _load_southwest_ood(args, getattr(args, 'attack_case', 'edge-case'))
    except Exception:
        return None
    if ood_test_np is None:
        return None
    # 构造 DataLoader
    ood_tensor = torch.tensor(ood_test_np, dtype=torch.float32)
    if ood_tensor.ndim == 3:
        ood_tensor = ood_tensor.unsqueeze(-1)
    ood_tensor = ood_tensor.permute(0, 3, 1, 2) / 255.0
    if getattr(args, 'dataset', 'cifar') == 'cifar':
        ood_tensor = (ood_tensor - 0.5) / 0.5
    ood_labels = torch.tensor(ood_test_lbl, dtype=torch.long)
    ood_dataset = torch.utils.data.TensorDataset(ood_tensor, ood_labels)
    loader = DataLoader(ood_dataset, batch_size=args.bs)
    # 评估目标类命中率
    net_g.eval()
    back_correct = 0
    back_num = 0
    attack_case = getattr(args, 'attack_case', 'edge-case')
    target_label = int(getattr(args, 'edge_target_label', 9)) if attack_case == 'edge-case' else 0
    for data, target in loader:
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        # 将所有样本视为目标标签
        target[:] = target_label
        back_num += data.shape[0]
        log_probs = net_g(data)
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        back_correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    back_accu = 100.00 * float(back_correct) / max(back_num, 1)
    return back_accu

def test_or_not(args, label):
    if args.attack_goal != -1:  # one to one
        if label == args.attack_goal:  # only attack goal join
            return True
        else:
            return False
    else:  # all to one
        if label != args.attack_label:
            return True
        else:
            return False
        
def add_trigger(args, image):
        if args.trigger == 'dba':
            pixel_max = 1
            image[:,args.triggerY+0:args.triggerY+2,args.triggerX+0:args.triggerX+2] = pixel_max
            image[:,args.triggerY+0:args.triggerY+2,args.triggerX+2:args.triggerX+5] = pixel_max
            image[:,args.triggerY+2:args.triggerY+5,args.triggerX+0:args.triggerX+2] = pixel_max
            image[:,args.triggerY+2:args.triggerY+5,args.triggerX+2:args.triggerX+5] = pixel_max
            save_img(image)
            return image
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
def save_img(image):
        img = image
        if image.shape[0] == 1:
            pixel_min = torch.min(img)
            img -= pixel_min
            pixel_max = torch.max(img)
            img /= pixel_max
            io.imsave('./save/test_trigger.png', img_as_ubyte(img.squeeze().cpu().numpy()))
        else:
            img = image.cpu().numpy()
            img = img.transpose(1, 2, 0)
            pixel_min = np.min(img)
            img -= pixel_min
            pixel_max = np.max(img)
            img /= pixel_max
            io.imsave('./save/test_trigger.png', img_as_ubyte(img))
