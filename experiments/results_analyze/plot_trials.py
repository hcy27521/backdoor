# plot_trials.py
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

# 连接 MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["backdoor"]
col = db["tm1:cifar:badnet:v2"]

# 取出所有 trial
docs = list(col.find())
print(f"共找到 {len(docs)} 条 trial")

# 转成 DataFrame
rows = []
for i, d in enumerate(docs, 1):
    res = d.get("result", {})
    train = res.get("train_stats", {})
    test = res.get("test_stats", {})
    bd = res.get("test_bd_stats", {})
    rows.append({
        "trial": i,
        "train_loss": train.get("train_eval_loss"),
        "train_acc": train.get("train_eval_acc"),
        "test_loss": test.get("test_eval_loss"),
        "test_acc": test.get("test_eval_acc"),
        "bd_loss": bd.get("test_bd_loss"),
        "bd_acc": bd.get("test_bd_acc"),
        "time": d.get("time"),
        "success": d.get("success"),
    })

df = pd.DataFrame(rows)
print(df.head())

# 画曲线
plt.figure(figsize=(10,6))
plt.plot(df["trial"], df["train_acc"], label="Train Acc")
plt.plot(df["trial"], df["test_acc"], label="Test Acc")
plt.plot(df["trial"], df["bd_acc"], label="Backdoor Acc")
plt.xlabel("Trial")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracies across Trials")
plt.grid(True)
plt.savefig("trial_results.png")
print("图已保存为 trial_results.png")
