#!/bin/bash
# 一键运行 EvilVGG11 训练/微调实验

# 基本参数设置
EXP_NAME="exp3_v1"          # 实验前缀，用于 MongoDB 和权重文件命名
DATASET="CIFAR10"              # 数据集 (需在 backdoor.dataset 里定义)
TRIGGER='checkerboard("bottomright",(1, 2),padding=1)'             # 后门触发器 (需在 backdoor.badnet.Trigger 支持)
EPOCHS=100                      # 训练轮数
LR=0.01                        # 学习率
DEVICE="cuda:0"                # 训练设备 (可改成 cpu)

# 权重文件保存目录
WEIGHTS_DIR="weights/evil"
mkdir -p "$WEIGHTS_DIR"

# 启动训练
python train_evilvgg11.py \
    -p $EXP_NAME \
    -d $DATASET \
    -t "$TRIGGER" \
    --epochs $EPOCHS \
    --learning_rate $LR \
    --device $DEVICE \
    --weights-path $WEIGHTS_DIR

