import os
import random
import torch
import torchvision
import numpy as np
from pymongo import MongoClient

# 你的项目模块
from backdoor import dataset
from backdoor import models  # 假设 EvilVGG11 在 models.CNN 中
from backdoor.badnet import BadNetDataPoisoning, Trigger
from backdoor.image_utils import ImageFormat
from backdoor.utils import totensor
from torchvision import transforms
from backdoor.training import Trainer
from backdoor.models import FCNN, CNN
from backdoor.search import Searchable, LogUniform, Choice, Uniform
from backdoor.handcrafted import FCNNBackdoor, CNNBackdoor

import argparse
parser = argparse.ArgumentParser(description="Train models on the chosen dataset with the chosen parameters")
parser.add_argument('task', nargs=1, choices=['clean', 'badnet', 'handcrafted'])
parser.add_argument('-p', '--prefix', type=str, required=True, help='Prefix for the experiment (weights, mongo). Should not include the task')
parser.add_argument('-d', '--dataset', type=str, help='Dataset to use', required=True)
parser.add_argument('-t', '--trigger', type=str, help='Trigger to use', required=True)
parser.add_argument('-n', '--trials', type=int, help='Number of trials to run', default=10)
parser.add_argument('-s', '--seed', type=int, help='Seed for random number generators', default=0)

parser.add_argument('--handcrafted-clean-weights', default=None, help='Required for handcrafted task. The weights file which should be attacked (defaults to {weights-path}/{prefix}:clean.pth)')
parser.add_argument('--attacked_model_path', default=None, help='Path to the attacked model file (if not provided, use default inside script)')
parser.add_argument('--mongo-url', default='mongodb://localhost:27017/', help="The URI of the MongoDB instance to save results to. Defaults to 'mongodb://localhost:27017/'")
parser.add_argument('--weights-path', default='weights', help='The folder in which to save weights files (must exist) - defaults to weights/')

parser.add_argument('--epochs', type=int, help='Number of epochs to train. Like other training options, this has no effect on handcrafted.', default=100)
parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.01)
parser.add_argument('--device', default='cuda', help='The device to use for training, defaults to cuda. Currently only affects handcrafted.', type=str)

parser.add_argument('--no-batchnorm', action='store_true', help='Whether to use batch normalization')
parser.add_argument('--use-wandb', action='store_true', help='Whether to use wandb')
parser.add_argument('--no-annealing', action='store_true', help='Whether to use annealing')
parser.add_argument('--no-dataaug', action='store_true', help='Whether to use data augmentation')
args = parser.parse_args()


# -----------------------
# 配置（根据需要改）
# -----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_dir = './weights'   # 保存权重的目录
os.makedirs(weights_dir, exist_ok=True)



# -----------------------
# 数据集：IMDBWiki
# -----------------------
ds = getattr(dataset, args.dataset)()
data = ds.get_data()   # 假设返回 {'train': (X_train, y_train), 'test': (X_test, y_test), ...}
# X 形状通常是 (N, H, W, C) 或 (N, C, H, W) 你项目中 ImageFormat.torch 会处理
backdoor_class = 7


trigger = Trigger.from_string(args.trigger)
badnet = BadNetDataPoisoning.always_backdoor(trigger, backdoor_class=backdoor_class)  
test_bd = badnet.apply(data['test'], poison_only=True)

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)


# -----------------------
# 通用重初始化函数
# -----------------------
def reinit_weights(module):
    """对 module 及其子模块做重初始化。尽量通用：Conv2d/Linear -> kaiming; BatchNorm -> reset."""
    for m in module.modules():
        # 优先使用类自带 reset_parameters
        if hasattr(m, 'reset_parameters'):
            try:
                m.reset_parameters()
            except Exception:
                pass

        # 针对常见层做手动初始化
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            try:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            except Exception:
                pass
            if getattr(m, 'bias', None) is not None:
                try:
                    torch.nn.init.constant_(m.bias, 0.)
                except Exception:
                    pass
        elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            # bn weight/bias 初始化为 1/0，running stats 清零/一
            if getattr(m, 'weight', None) is not None:
                try:
                    torch.nn.init.constant_(m.weight, 1.0)
                except Exception:
                    pass
            if getattr(m, 'bias', None) is not None:
                try:
                    torch.nn.init.constant_(m.bias, 0.0)
                except Exception:
                    pass
            # reset running stats
            if hasattr(m, 'running_mean'):
                m.running_mean.zero_()
            if hasattr(m, 'running_var'):
                m.running_var.fill_(1)

# -----------------------
# 加载攻击者给的文件，区分两种情况
# -----------------------
def load_attacked_model(path, model_cls=None, model_ctor_kwargs=None):
    """
    尝试 torch.load(path).
    如果 load 得到的是 nn.Module -> 返回该实例 (caller 可 reinit)
    如果 load 得到的是 state_dict -> 需要 model_cls 提供以构造实例再 load_state_dict
    """
    obj = torch.load(path, map_location='cpu')  # 先 load 到 cpu，之后再 to(device)
    if isinstance(obj, torch.nn.Module):
        return obj  # caller 可以直接用 .apply(reinit_weights) 或重新实例化
    elif isinstance(obj, dict):
        # 有时保存的 dict 里可能包含 'state_dict' 键
        if 'state_dict' in obj:
            state = obj['state_dict']
        else:
            state = obj
        if model_cls is None:
            raise ValueError("加载到 state_dict，但没有提供 model_cls 来实例化模型。")
        kwargs = model_ctor_kwargs or {}
        model = model_cls((ds.n_channels, *ds.image_shape), ds.n_classes, **kwargs)
        model.load_state_dict(state)
        return model
    else:
        raise ValueError("不支持的模型文件内容。")

# -----------------------
# 训练函数：对给定 attacked model 进行 reinit + training
# -----------------------
'''def retrain_from_attacked(attacked_path, out_prefix, n_runs=1, epochs=20, lr=0.1, batch_size=256,
                          model_cls_attacked=None, model_ctor_kwargs=None):
    """
    attacked_path: 攻击者递交的文件路径（可能是完整模型或 state_dict）
    model_cls_attacked: 如果 attacked_path 是 state_dict，你必须提供构造该架构的类（如 models.CNN.EvilVGG11）
    out_prefix: 用于保存/记录前缀
    """
    for run_i in range(n_runs):
        print(f'=== retrain run {run_i} ===')
        # 加载（或构造）被攻击模型架构实例
        mdl = load_attacked_model(attacked_path, model_cls=model_cls_attacked, model_ctor_kwargs=model_ctor_kwargs)

        # 确保是同一架构对象（保留架构），对权重重初始化
        reinit_weights(mdl)

        # 移到 device
        mdl.to(device)

        # 构造 Trainer（你的项目 Trainer 接口）
        trainer = Trainer(mdl, optimizer=torch.optim.SGD,
                          optimizer_params=dict(lr=lr),
                          use_wandb=False)

        if epochs > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optim, T_max=epochs)

        history = []
        for epoch in range(epochs):
            print(f'Run {run_i} Epoch {epoch} LR={trainer.optim.param_groups[0]["lr"]:.6f}')
            trainer.train_epoch(*data['train'], bs=batch_size, progress_bar=True, shuffle=True,
                                tfm=None)  # 如果你有 transform，传进去

            # 评估
            with torch.no_grad():
                train_stats = trainer.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
                test_stats = trainer.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)

                # 对带触发的 test_bd 评估：若你希望触发样本的标签是 attacker_target_label，
                # 则 test_bd.y 要非空；否则 evaluate_epoch 可能接受 (X,y) 或 一个 Dataset-like
                test_bd_stats = trainer.evaluate_epoch(test_bd, bs=512, name='test_bd', progress_bar=False)
                # 带触发但用 clean label 评估（论文里也做过）
                if test_bd.y is not None:
                    test_bd_neg_stats = trainer.evaluate_epoch(test_bd.X, data['test'][1], bs=512,
                                                               name='test_bd_neg', progress_bar=False)
                else:
                    test_bd_neg_stats = None

            stats = {'train_stats': train_stats, 'test_stats': test_stats,
                     'test_bd_stats': test_bd_stats, 'test_bd_neg_stats': test_bd_neg_stats}
            history.append(stats)
            print('  test acc (clean):', test_stats.get('accuracy', test_stats))
            print('  test acc (triggered):', test_bd_stats.get('accuracy', test_bd_stats))

            if epochs > 0:
                scheduler.step()

        # 保存模型
        save_path = os.path.join(weights_dir, f'{out_prefix}:reinit_{random.randrange(16**5+1, 16**6):x}.pth')
        torch.save(mdl, save_path)
        print('Saved re-trained model to', save_path)

        db = MongoClient(args.mongo_url)['backdoor'][f'{out_prefix}:clean']
        db.insert_one({'args': vars(args), 'history': history, 'weights': save_path, 'stats': stats})

# -----------------------
# 使用示例
# -----------------------
# 情形 A：attacked model 文件里保存的是完整的 nn.Module（你直接传路径）
try:
    retrain_from_attacked(attacked_model_path, out_prefix='tm3_arch_attack', n_runs=1, epochs=30, lr=0.01,
                          batch_size=256, model_cls_attacked=None, model_ctor_kwargs={'batch_norm': True})
except Exception as e:
    print('直接 load 失败或文件是 state_dict：', e)
    # 情形 B：若文件实际上是 state_dict，需要提供 model_cls_attacked，比如 EvilVGG11
    retrain_from_attacked(attacked_model_path, out_prefix='tm3_arch_attack', n_runs=1, epochs=30, lr=0.01,
                          batch_size=256, model_cls_attacked=models.CNN.EvilVGG11,
                          model_ctor_kwargs={'batch_norm': True})
'''
def format_stats(stats):
    keylen = max([len(k) for k in stats.keys() if k != 'history'])
    print(f"{' '.rjust(keylen)}  LOSS   ACC")
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            print(f"{k.rjust(keylen)} {v}")
        elif isinstance(v, dict) and len(v) == 2:
            subkeys = v.keys()
            loss = v[[sk for sk in subkeys if 'loss' in sk][0]]
            acc = v[[sk for sk in subkeys if 'acc' in sk][0]]
            print(f"{k.ljust(keylen)} {loss:.4f} {acc:.4f}")

######################
#######替换分类层#####
######################
def replace_last_layer_cnn(model, num_classes):
    """支持 CNN + FCNN 的分类头替换"""
    if hasattr(model, "fcnn_module") and hasattr(model.fcnn_module, "fc_layers"):
        last = model.fcnn_module.fc_layers[-1]
        if isinstance(last, torch.nn.Linear):
            in_features = last.in_features
            model.fcnn_module.fc_layers[-1] = torch.nn.Linear(in_features, num_classes).to(last.weight.device)
            print(f"✅ 替换分类头: {in_features} → {num_classes}")
            return model

    raise ValueError("未找到可替换的分类头，请检查模型结构")



##########################
##### Clean Training #####
##########################

# Transforms to improve performance
if not args.no_dataaug:
    transform = transforms.Compose([
        transforms.RandomCrop(ds.image_shape[0], padding=4),
        transforms.RandomHorizontalFlip(),
    ])
else:
    transform = transforms.Compose([])

def train_clean(attacked_path, prefix, n, model_cls_attacked=None, model_ctor_kwargs=None):
    def train_model_clean():
        print('Training CLEAN model')

        model_clean = load_attacked_model(attacked_path, model_cls=model_cls_attacked, model_ctor_kwargs=model_ctor_kwargs)
        # print(torchsummary.summary(model_clean, (ds.n_channels, *ds.image_shape)))

        reinit_weights(model_clean)
        #替换最后的分类头，让输出维度匹配 IMDBWiki 的类别数
        replace_last_layer_cnn(model_clean, ds.n_classes)
        model_clean = model_clean.to(device)

        history = []

        t = Trainer(model_clean, optimizer=torch.optim.SGD,
            optimizer_params=dict(lr=args.learning_rate, momentum=0.9, weight_decay=5e-4),
            use_wandb=args.use_wandb)
        if not args.no_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t.optim, T_max=args.epochs)
        for i in range(args.epochs):
            print(f'* Epoch {i} - LR={t.optim.param_groups[0]["lr"]:.5f}')
            t.train_epoch(*data['train'], bs=256, progress_bar=False, shuffle=True, tfm=transform)

            # Evaluate on both datasets
            train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
            test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
            test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)

            # Performance on backdoored data with clean labels (for threat model 3)
            test_bd_neg_stats = t.evaluate_epoch(test_bd.X, data['test'].y, bs=512, name='test_bd_neg', progress_bar=False)

            stats = {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats, 'test_bd_neg_stats': test_bd_neg_stats}
            format_stats(stats)
            history.append(stats)

            # Finish epoch, update learning rate
            if not args.no_annealing:
                scheduler.step()

        save_path = f'{args.weights_path}/{prefix}:clean_{random.randrange(16**5+1, 16**6):x}.pth'
        torch.save(model_clean, save_path)

        # Save stats to Mongo
        db = MongoClient(args.mongo_url)['backdoor'][f'{prefix}:clean']
        db.insert_one({'args': vars(args), 'history': history, 'weights': save_path, 'stats': stats})

    for i in range(n):
        print('Training model', i)
        train_model_clean()


##### BadNets Training #####
# Set up a function we can random search over
def train_badnet(attacked_path, prefix, n, model_cls_attacked=None, model_ctor_kwargs=None):
    def train_model_badnet(poison_proportion):
        print('Training with poison proportion of', poison_proportion)

        # 1. load attacked architecture (or build new if attacked_path is None)
        model = load_attacked_model(attacked_path, model_cls=model_cls_attacked, model_ctor_kwargs=model_ctor_kwargs)

        # 2. reinit all weights (defender policy)
        reinit_weights(model)

        # 3. replace last layer to match ds.n_classes
        replace_last_layer_cnn(model, ds.n_classes)

        # 4. move to device
        model = model.to(device)

        history = []
        t = Trainer(model, optimizer=torch.optim.SGD,
                    optimizer_params=dict(lr=args.learning_rate, momentum=0.9, weight_decay=5e-4),
                    use_wandb=args.use_wandb)
        if not args.no_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t.optim, T_max=args.epochs)

        for i in range(args.epochs):
            print(f'* Epoch {i} - LR={t.optim.param_groups[0]["lr"]:.5f}')

            # prepare train data (same as original): transform then random-poison
            X_train, y_train = data['train']
            X_train_aug = transform(ImageFormat.torch(X_train, tensor=True))

            # Apply BadNet poisoning (on-the-fly)
            X_train_aug, y_train_aug = badnet.apply_random_sample((X_train_aug, y_train), poison_proportion)

            t.train_epoch(X_train_aug, y_train_aug, bs=256, progress_bar=False, shuffle=True)

            # Evaluate
            train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
            test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
            test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
            test_bd_neg_stats = t.evaluate_epoch(test_bd.X, data['test'].y, bs=512, name='test_bd_neg', progress_bar=False)

            rec = {'train_stats': train_stats, 'test_stats': test_stats,
                   'test_bd_stats': test_bd_stats, 'test_bd_neg_stats': test_bd_neg_stats}
            history.append(rec)
            format_stats(rec)

            if not args.no_annealing:
                scheduler.step()

        weights = f'{args.weights_path}/{prefix}:badnet_{poison_proportion}_{random.randrange(16**5+1, 16**6):x}.pth'
        torch.save(model, weights)

        return {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats,
                'weights': weights, 'history': history, 'args': vars(args)}

    db = MongoClient(args.mongo_url)['backdoor'][f'{prefix}:badnet']
    train_model_badnet = Searchable(train_model_badnet, db)
    train_model_badnet.random_search([LogUniform(0.001, 0.1)], {}, trials=n)


def train_handcrafted(attacked_path, prefix, n, model_cls_attacked=None, model_ctor_kwargs=None):
    def train_model_handcrafted(**kwargs):
        # If no clean weights provided, default path
        if args.handcrafted_clean_weights is None:
            args.handcrafted_clean_weights = f'{args.weights_path}/{prefix}:clean.pth'

        # Load base model (from attacked_path) then reinit and replace head
        model = load_attacked_model(attacked_path, model_cls=model_cls_attacked, model_ctor_kwargs=model_ctor_kwargs)
        reinit_weights(model)
        replace_last_layer_cnn(model, ds.n_classes)
        model = model.to(args.device)

        # For handcrafted insertion we pick representative samples from train
        ixs = np.random.permutation(np.arange(len(data['train'][0])))[:512]
        X_batch_clean = data['train'][0][ixs]
        y_batch_clean = data['train'][1][ixs]
        X_batch_bd, y_batch_bd = badnet.apply(data['train'], poison_only=True)
        X_batch_bd = X_batch_bd[ixs]
        y_batch_bd = y_batch_bd[ixs]

        handcrafted = CNNBackdoor(model, device=args.device)
        handcrafted.insert_backdoor(X_batch_clean, y_batch_clean, X_batch_bd, **kwargs, enforce_min_separation=False)

        # Evaluate (no further training)
        t = Trainer(model, use_wandb=False, device=args.device)
        train_stats = t.evaluate_epoch(*data['train'], bs=512, name='train_eval', progress_bar=False)
        test_stats = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
        test_bd_stats = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
        test_bd_neg_stats = t.evaluate_epoch(test_bd.X, data['test'].y, bs=512, name='test_bd_neg', progress_bar=False)

        weights = f'{args.weights_path}/{prefix}:handcrafted_{random.randrange(16**5+1, 16**6):x}.pth'
        torch.save(model, weights)

        stats = {'train_stats': train_stats, 'test_stats': test_stats, 'test_bd_stats': test_bd_stats,
                 'test_bd_neg_stats': test_bd_neg_stats, 'weights': weights, 'args': vars(args)}
        format_stats(stats)
        return stats

    db = MongoClient(args.mongo_url)['backdoor'][f'{prefix}:handcrafted']
    train_model_handcrafted = Searchable(train_model_handcrafted, db)

    train_model_handcrafted.random_search(
        [],
        dict(
            neuron_selection_mode='acc',
            acc_th=0.014922959761277917,
            num_to_compromise=4,
            min_separation=0.99,
            guard_bias_k=1.0111481826673672,
            backdoor_class=backdoor_class,
            target_amplification_factor=23.98123235523619,
            max_separation_boosting_rounds=10,
            n_filters_to_compromise=3,
            conv_filter_boost_factor=1.351463725536423
        ),
        trials=n,
        on_error='return',
        seed=args.seed
    )


print(args)
if args.task[0] == 'clean':
    train_clean(args.attacked_model_path, args.prefix, args.trials, model_cls_attacked=None, model_ctor_kwargs={'batch_norm': True})
elif args.task[0] == 'badnet':
    train_badnet(args.attacked_model_path, args.prefix, args.trials, model_cls_attacked=None, model_ctor_kwargs={'batch_norm': True})
elif args.task[0] == 'handcrafted':
    train_handcrafted(args.prefix, args.trials)
