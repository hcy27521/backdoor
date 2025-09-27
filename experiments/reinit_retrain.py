import os
import random
import torch
import numpy as np
from pymongo import MongoClient

# 你的项目模块
from backdoor.dataset import IMDBWiki
from backdoor import models  # 假设 EvilVGG11 在 models.CNN 中
from backdoor.badnet import BadNetDataPoisoning, Trigger
from backdoor.image_utils import ImageFormat
from backdoor.utils import totensor
from backdoor.training import Trainer

# -----------------------
# 配置（根据需要改）
# -----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_dir = './weights'   # 保存权重的目录
os.makedirs(weights_dir, exist_ok=True)

# 例如 attacker 给你的模型文件：
attacked_model_path = '/home/wyl/backdoor/experiments/weights/cifar_badnet/tm1_cifar_badnets_0.0013292918943162162.pth'  # 攻击者给的模型文件（示例）
# 若你有多个模型（badnets/handcrafted/arch），对每个调用下面的流程


# -----------------------
# 数据集：IMDBWiki
# -----------------------
ds = IMDBWiki()
data = ds.get_data()   # 假设返回 {'train': (X_train, y_train), 'test': (X_test, y_test), ...}
# X 形状通常是 (N, H, W, C) 或 (N, C, H, W) 你项目中 ImageFormat.torch 会处理
backdoor_class = 7

trigger = Trigger.from_string("checkerboard('bottomleft', (3, 3), colours=(255, 0))")
badnet = BadNetDataPoisoning.always_backdoor(trigger, backdoor_class=backdoor_class)  
test_bd = badnet.apply(data['test'], poison_only=True)


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
def retrain_from_attacked(attacked_path, out_prefix, n_runs=1, epochs=20, lr=0.1, batch_size=256,
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
