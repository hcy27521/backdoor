'''在CIFAR10数据集上进行后门攻击（Backdoor Attack）的实验'''
from torch import nn
import torch
import torchvision
import numpy as np
import random
import backdoor
import os

from pymongo import MongoClient

from typing import Tuple

from backdoor.models import FCNN, CNN
from backdoor import dataset
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning, Trigger
from backdoor.image_utils import ImageFormat, ScikitImageArray
from backdoor.handcrafted import FCNNBackdoor, CNNBackdoor
from backdoor.search import Searchable, LogUniform, Choice, Uniform

from pymongo.mongo_client import MongoClient

import torchsummary

#### OPTIONS
USE_BATCHNORM = True #适用批归一化
USE_ANNEALING = True #使用学习率退火（余弦退火）
DATA_AUGMENTATION = True
N_EPOCHS = 50
LEARNING_RATE = 0.1
backdoor_class = 6 #后门目标类别
TRIGGER = "checkerboard('bottomright', (1, 2), padding=1)" #触发器定义字符串：一个位于右下角、1行2列、带1像素填充的棋盘格

#####控制哪些实验要运行的开关
TRAIN_CLEAN = False
TRAIN_BADNETS = False
TRAIN_HANDCRAFTED = True
#####

use_wandb = False
if use_wandb:
    import wandb
    wandb.init(project='backdoor', entity='mxbi', 
    config={'batch_norm': USE_BATCHNORM, 'data_augmentation': DATA_AUGMENTATION, 'learning_rate': LEARNING_RATE, 'n_epochs': N_EPOCHS, 'trigger': TRIGGER}
    )

ds = dataset.CIFAR10()
data = ds.get_data()

#设置随机种子，确保随机数的可重复性
np.random.seed(0)
torch.random.manual_seed(0)

# Construct the trigger function & dataset

trigger = Trigger.from_string(TRIGGER)
#创建BadNet数据投毒对象，该对象会将任何输入图像都加上触发器并标记为backdoor_class
badnet = BadNetDataPoisoning.always_backdoor(trigger, backdoor_class=backdoor_class)
#对测试集应用后门攻击，生成一个后门测试集（poison_only=True表示只生成带触发器的样本，不混合干净样本）
test_bd = badnet.apply(data['test'], poison_only=True)

##########################
##### Clean Training #####
##########################

# Transforms to improve performance
from torchvision import transforms
if DATA_AUGMENTATION:
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
else:
    transform = transforms.Compose([])

# From Table X in Handcrafted paper
# NOTE: This model is slightly different to the one in the paper. We have an extra maxpool layer because this is required by our handcrafted implementation
if TRAIN_CLEAN:
    #创建VGG11模型，输入尺寸为CIFAR-10（通道数，高，宽），输出10类，根据配置决定是否使用批归一化
    model_clean = CNN.VGG11((ds.n_channels, *ds.image_shape), 10, batch_norm=USE_BATCHNORM) 
    # 打印模型的结构摘要，显示各层参数
    print(torchsummary.summary(model_clean, (ds.n_channels, *ds.image_shape)))

    t = Trainer(model_clean, optimizer=torch.optim.SGD, optimizer_params=dict(lr=LEARNING_RATE), use_wandb=use_wandb)
    #如果使用学习率退火，则创建余弦退火学习率调度器
    if USE_ANNEALING:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t.optim, T_max=N_EPOCHS)
    for i in range(N_EPOCHS):
        print(f'* Epoch {i}')
        t.train_epoch(*data['train'], bs=256, progress_bar=False, shuffle=True, tfm=transform)

        # 在每个epoch后，在训练集、干净测试集、后门测试集上进行评估
        train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
        test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
        test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
        print('Training set performance:', train_stats)
        print('Test set performance:', test_stats)
        print(test_bd_stats)

        final_test_performance_clean = test_stats['test_eval_acc']
        final_test_bd_performance_clean = test_bd_stats['test_bd_acc']

        # Finish epoch, update learning rate
        # 如果是退火回合，更新学习率
        if USE_ANNEALING:
            scheduler.step()
        print("Learning rate:", t.optim.param_groups[0]['lr'])

    torch.save(model_clean, '/home/wyl/backdoor/experiments/weights/tm1_cifar_clean.pth')

##### BadNets Training #####

# 定义一个函数，用于训练单个BadNet模型。它接受一个参数：毒化比例（poison_proportion）
def train_model_badnet(poison_proportion):
    print('Training with poison proportion of', poison_proportion)

    model = CNN.VGG11((ds.n_channels, *ds.image_shape), 10, batch_norm=USE_BATCHNORM) 
    # print(torchsummary.summary(model, (ds.n_channels, *ds.image_shape)))

    history = [] # 用于记录每个epoch的历史

    t = Trainer(model, optimizer=torch.optim.SGD, optimizer_params=dict(lr=LEARNING_RATE), use_wandb=use_wandb)
    if USE_ANNEALING:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t.optim, T_max=N_EPOCHS)
    for i in range(N_EPOCHS):
        print(f'* Epoch {i}')

        # We perform the transform here before the training process, so that we can then apply the trigger afterwards
        # 1. 获取原始训练数据
        X_train, y_train = data['train']
        # 2. 对原始数据应用数据增强变换（随机裁剪/翻转）
        X_train_aug = transform(ImageFormat.torch(X_train, tensor=True))

        # Apply the backdoor
        # 3. 【关键步骤】对增强后的数据应用后门攻击
        #    badnet.apply_random_sample 会以 poison_proportion 的概率随机选择样本，为其添加触发器并修改标签为 backdoor_class
        #    其余样本保持不变。这样就得到了一个混合了干净样本和毒化样本的训练批次
        X_train_aug, y_train_aug = badnet.apply_random_sample((X_train_aug, y_train), poison_proportion)

        t.train_epoch(X_train_aug, y_train_aug, bs=256, progress_bar=False, shuffle=True)

        # Evaluate on both datasets
        train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
        test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
        test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)

        history.append({'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats})
        print(history[-1])

        # Finish epoch, update learning rate
        if USE_ANNEALING:
            scheduler.step()
        print("Learning rate:", t.optim.param_groups[0]['lr'])

    weights = "/home/wyl/backdoor/experiments/weights/badnet_final.pth"
    torch.save(model, weights)

    return {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats, 'weights': weights, 'history': history}

if TRAIN_BADNETS:
    # 连接到MongoDB数据库，指定数据库和集合（collection）
    db = MongoClient('mongodb://localhost:27017/')['backdoor']['tm1:cifar:badnet:v2']
    # 用Searchable包装训练函数，使其具备超参数搜索和结果保存到数据库的功能
    train_model_badnet = Searchable(train_model_badnet, db)

    # 执行随机搜索。
    # 参数1: [LogUniform(0.0001, 0.1)] 定义搜索空间：毒化比例，在对数尺度上从0.01%到10%之间随机选择。
    # 参数2: {} 没有其他需要搜索的参数。
    # 参数3: trials=100 尝试100组不同的超参数。
    train_model_badnet.random_search([LogUniform(0.0001, 0.1)], {}, trials=100)


##### Handcrafted Training #####
# 定义一个函数，用于执行手工后门注入。它接受许多超参数 (**kwargs)
def train_model_handcrafted(**kwargs):
    model = torch.load('/home/wyl/backdoor/experiments/weights/tm1_cifar_clean.pth', map_location='cpu')
    DEVICE = torch.device("cuda:1")
    model = model.to(DEVICE)

    # 准备一小批干净数据和一小批后门数据，用于指导后门注入过程
    X_batch_clean = data['train'][0][:128]
    y_batch_clean = data['train'][1][:128]
    X_batch_bd, y_batch_bd = badnet.apply(data['train'], poison_only=True)
    X_batch_bd = X_batch_bd[:128]
    y_batch_bd = y_batch_bd[:128]

    # 创建CNNBackdoor对象，用于对CNN模型进行手工后门注入
    handcrafted = CNNBackdoor(model)
    # 【核心操作】执行后门注入。
    # 该方法会分析模型对干净样本和后门样本的响应，然后直接修改模型的特定参数（如权重、偏置），
    # 植入后门行为，使得模型在看到触发器时输出目标类别。
    handcrafted.insert_backdoor(X_batch_clean, y_batch_clean, X_batch_bd, **kwargs, enforce_min_separation=False)

    # 创建训练器（主要用于评估，因为模型参数已被直接修改，不需要再训练）
    t = Trainer(model, use_wandb=False) # We don't actually train, just for evaluation

    train_stats = t.evaluate_epoch(*data['train'], bs=128, name='train_eval', progress_bar=False)
    test_stats = t.evaluate_epoch(*data['test'], bs=128, name='test_eval', progress_bar=False)
    test_bd_stats = t.evaluate_epoch(*test_bd, bs=128, name='test_bd', progress_bar=False)

    weights = "/home/wyl/backdoor/experiments/weights/handcrafted_final.pth"
    torch.save(model, weights)


    stats = {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats, 'weights': weights}
    print(stats)
    # 每次释放
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return stats
    


if TRAIN_HANDCRAFTED:
    db = MongoClient('mongodb://localhost:27017/')['backdoor']['tm1:cifar:handcrafted:v2']
    train_model_handcrafted = Searchable(train_model_handcrafted, db)

    # 执行随机搜索。
    # 参数1: [] 没有位置参数。
    # 参数2: 一个庞大的字典，定义了手工后门注入方法的所有可调超参数及其搜索空间。
    #        例如：neuron_selection_mode（神经元选择模式）、acc_th（准确率阈值）、num_to_compromise（要破坏的神经元数量）等等。
    #        这些参数精细地控制了后门注入的过程和行为。
    # 参数3: trials=500 尝试500组不同的超参数组合。
    # 参数4: on_error='return' 如果某次运行出错，记录错误并继续下一组，而不是中断整个搜索。
    # 参数5: seed=30 为随机搜索设置特定的种子，确保可重复性。
    train_model_handcrafted.random_search([], 
    dict(
        neuron_selection_mode='acc',
        acc_th=Uniform(0, 0.05),
        num_to_compromise=LogUniform(1, 10, integer=True),
        min_separation=Choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        guard_bias_k=Uniform(0.5, 2),
        backdoor_class=6,
        target_amplification_factor=LogUniform(1, 50),
        max_separation_boosting_rounds=10,
        n_filters_to_compromise=LogUniform(1, 10, integer=True),
        conv_filter_boost_factor=LogUniform(0.1, 5)
    ),
    trials=10,
    on_error='return',
    seed=30
    )
