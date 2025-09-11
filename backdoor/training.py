import torch
import numpy as np
from tqdm import tqdm

from .utils import totensor, tonp
from .image_utils import ImageFormat

import swanlab as wandb

class Trainer:
    def __init__(self, model, criterion=torch.nn.CrossEntropyLoss(reduction='mean'), 
                optimizer=torch.optim.SGD, optimizer_params={'lr': 0.01}, device='cuda:1', use_wandb=True,
                convert_image_format=True):
        """self.model = model.to(device)
        self.device = device"""
        self.device = torch.device(device)  # 确保是 torch.device
        self.model = model.to(self.device)

        self.criterion = criterion  #损失函数
        self.optim = optimizer(self.model.parameters(), **optimizer_params)
        self.convert_image_format = convert_image_format # 是否转换图像格式标志


        # whether to enable wandb logging
        self.wandb = use_wandb

        if 'lr' in optimizer_params and self.wandb:
            wandb.log({'lr': optimizer_params['lr']})

    def set_learning_rate(self, lr):
        # 动态设置学习率
        for g in self.optim.param_groups:
            g['lr'] = lr
        if self.wandb:
            wandb.log({'lr': lr})

    def get_mean_gradients(self):
        # 计算模型参数梯度的平均值
        abs_gradient_sum = 0
        abs_gradient_count = 0
        for param in self.model.parameters():
            abs_grad = torch.abs(param.grad)
            abs_gradient_sum += torch.sum(abs_grad)
            abs_gradient_count += torch.prod(abs_grad)
        return abs_gradient_sum / abs_gradient_count

    def batch_inference(self, X):
        """
        对单个批次进行推理。
        返回预测的Torch张量。
        """
        self.model.eval()

        if self.convert_image_format:
            X = ImageFormat.torch(X) # 转换图像格式为PyTorch标准格式

        return self.model(totensor(X, device=self.device))

    def inference(self, X, batch_size=64):
        """
        对大型数据集进行推理，分成多个批次。
        返回预测的NumPy数组。
        """

        self.model.eval()

        if self.convert_image_format:
            X = ImageFormat.torch(X)

        n_batches = int(np.ceil(len(X) / batch_size)) # 计算批次数量
        outputs = []
        for i_batch in range(n_batches):
            x_batch = totensor(X[i_batch*batch_size:(i_batch+1)*batch_size], device=self.device)
            outputs.append(tonp(self.model(x_batch)))
        return np.concatenate(outputs)

    def train_epoch(self, X, y, sample_weights=None, bs=64, shuffle=False, name='train', progress_bar=True, tfm=None):
        # 训练一个epoch
        assert len(X) == len(y), "X and y must be the same length"    # 确保数据和标签长度一致
        self.model.train()
        n_batches = int(np.ceil(len(X) / bs))

        # Convert image format to conform to training process (if necessary)
        if self.convert_image_format:
            X = ImageFormat.torch(X)

        # Randomly shuffle if required
        if shuffle:
            shuffle_ixs = np.random.permutation(np.arange(len(X)))  # 生成随机索引
            X = X[shuffle_ixs]
            y = y[shuffle_ixs]

            if sample_weights is not None:
                sample_weights = sample_weights[shuffle_ixs] # 重新排列样本权重

        if 'cuda' in self.device:
            LongTensor = torch.cuda.LongTensor
        else:
            LongTensor = torch.LongTensor

         # 主训练循环
        for i_batch in (tqdm(range(n_batches)) if progress_bar else range(n_batches)):
            x_batch = totensor(X[i_batch*bs:(i_batch+1)*bs], device=self.device)
            y_batch = totensor(y[i_batch*bs:(i_batch+1)*bs], device=self.device, type=int)

            if tfm:
                x_batch = tfm(x_batch) # 应用数据增强变换

            # print(x_batch.min(), x_batch.max())

            self.optim.zero_grad() # 梯度清零
            outputs = self.model(x_batch) # 前向传播

            if sample_weights is not None:
                if self.criterion.reduction != 'none':
                    raise ValueError("Trying to use `sample_weights` with a reduced criterion. Use reduction='none' when specifying the criterion to allow sample weights to be applied.")

                loss = self.criterion(outputs, y_batch.type(LongTensor))

                w_batch = totensor(sample_weights[i_batch*bs:(i_batch+1)*bs], device=self.device)
                loss = (loss * w_batch).mean()
            else:
                loss = self.criterion(outputs, y_batch.type(LongTensor))
                # print(loss)
                # If the user specifies criterion with reduction=none, this means it can be used both with and without sample weights.
                # 如果用户指定了reduction=none的准则，则取平均值
                if self.criterion.reduction == 'none':
                    loss = loss.mean()

            # Accuracy measurement
            outputs_cpu = tonp(outputs)
            batch_acc = (y[i_batch*bs:(i_batch+1)*bs] == outputs_cpu.argmax(1)).mean()

            if torch.isnan(loss): 
                print('WARN: dropping batch with nan loss')
                continue

            try:
                loss.backward()
            except RuntimeError:
                print(loss)
                print(x_batch.mean(), x_batch.std())
                raise
            gradient_size = self.get_mean_gradients()
            self.optim.step()
            
            if self.wandb:
                wandb.log({f"{name}_batch_loss": loss, f"{name}_batch_lma_gradient": np.log(tonp(gradient_size)), f"{name}_batch_acc": batch_acc})

    def evaluate_epoch(self, X, y, bs=64, name='eval', progress_bar=True):
        assert len(X) == len(y), "X and y must be the same length"
        self.model.eval()
        n_batches = int(np.ceil(len(X) / bs))

        if self.convert_image_format:
            X = ImageFormat.torch(X)

        LongTensor = torch.cuda.LongTensor if 'cuda' in self.device else torch.LongTensor

        total_loss = 0.
        total_acc = 0.
        for i_batch in (tqdm(range(n_batches)) if progress_bar else range(n_batches)):
            x_batch = totensor(X[i_batch*bs:(i_batch+1)*bs], device=self.device)
            y_batch = totensor(y[i_batch*bs:(i_batch+1)*bs], device=self.device, type=int)

            outputs = self.model(x_batch)

            loss = self.criterion(outputs, y_batch.type(LongTensor))
            # If the user specifies criterion with reduction=none, allow eval to still work.
            if self.criterion.reduction == 'none':
                    loss = loss.mean()

            # Add loss on the batch to the accumulator
            total_loss += tonp(loss) * len(x_batch)

            # Measure accuracy on the batch and add to accumulator
            outputs_cpu = tonp(outputs)
            total_acc += (y[i_batch*bs:(i_batch+1)*bs] == outputs_cpu.argmax(1)).sum()

        # Get summary statistics over the whole epoch
        epoch_metrics = {f"{name}_loss": total_loss / len(X), f"{name}_acc": total_acc / len(X)}
        if self.wandb:
            wandb.log(epoch_metrics)

        return epoch_metrics