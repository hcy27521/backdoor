import matplotlib.pyplot as plt
from pymongo import MongoClient
import pandas as pd

# 1. 连接 MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["backdoor"]  # 数据库名称
collection = db["tm1:cifar:badnet:v2"]  # 你之前看到的集合

# 2. 读取所有 trial 文档
docs = list(collection.find().sort("_id", 1))

if not docs:
    print("没有找到任何 trial 数据！请确认训练已写入数据库。")
    exit()

# 3. 转换为 DataFrame
records = []
for i, d in enumerate(docs):
    stats = d.get("stats", {})  # 每个 trial 训练完成时的结果
    records.append({
        "trial": i + 1,
        "train_loss": stats.get("train_stats", {}).get("train_eval_loss", None),
        "train_acc": stats.get("train_stats", {}).get("train_eval_acc", None),
        "test_loss": stats.get("test_stats", {}).get("test_eval_loss", None),
        "test_acc": stats.get("test_stats", {}).get("test_eval_acc", None),
        "bd_loss": stats.get("test_bd_stats", {}).get("test_bd_loss", None),
        "bd_acc": stats.get("test_bd_stats", {}).get("test_bd_acc", None),
    })

df = pd.DataFrame(records)
print(df.head())

# 4. 画曲线
plt.figure(figsize=(12, 8))

# Acc 曲线
plt.subplot(2, 1, 1)
plt.plot(df["trial"], df["train_acc"], label="Train Acc")
plt.plot(df["trial"], df["test_acc"], label="Test Acc")
plt.plot(df["trial"], df["bd_acc"], label="Backdoor Acc")
plt.xlabel("Trial")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves")
plt.legend()
plt.grid(True)

# Loss 曲线
plt.subplot(2, 1, 2)
plt.plot(df["trial"], df["train_loss"], label="Train Loss")
plt.plot(df["trial"], df["test_loss"], label="Test Loss")
plt.plot(df["trial"], df["bd_loss"], label="Backdoor Loss")
plt.xlabel("Trial")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("trial_curves.png", dpi=200)
plt.show()

