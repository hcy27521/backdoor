# analyze_badnet_results.py
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

# 文件路径
csv_file = "gtsb:badnet.csv"
img_file = "badnet_results.png"

# 读取实验结果
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"未找到 {csv_file}，请确认路径正确！")

df = pd.read_csv(csv_file)

print(f"共找到 {len(df)} 条 trial")
print(df.head())

# 将字符串形式的 dict 转成真正的 dict
df["result"] = df["result"].apply(ast.literal_eval)

# 从 result 里提取干净/后门精度和 poison_rate
df["clean_acc"] = df["result"].apply(lambda x: x["test_stats"]["test_eval_acc"])
df["bd_acc"] = df["result"].apply(lambda x: x["test_bd_stats"]["test_bd_acc"])
df["poison_rate"] = df["args"].apply(lambda x: float(ast.literal_eval(x)[0]))

# 计算综合指标
df["score"] = df["clean_acc"] * df["bd_acc"]

# 找出最优 trial
best_trial = df.loc[df["score"].idxmax()]
print("\n=== 最优 Trial ===")
print(best_trial[["poison_rate", "clean_acc", "bd_acc", "score"]])

# 绘制曲线：poison_rate vs clean_acc / bd_acc
plt.figure(figsize=(8,6))
plt.scatter(df["poison_rate"], df["clean_acc"], label="Clean Acc", alpha=0.7)
plt.scatter(df["poison_rate"], df["bd_acc"], label="Backdoor Acc", alpha=0.7)

# 标记最优 trial
plt.scatter(best_trial["poison_rate"], best_trial["clean_acc"],
            color="blue", edgecolor="black", s=120, marker="o", label="Best Clean")
plt.scatter(best_trial["poison_rate"], best_trial["bd_acc"],
            color="red", edgecolor="black", s=120, marker="o", label="Best Backdoor")

plt.xlabel("Poison Rate")
plt.ylabel("Accuracy")
plt.title("Clean Accuracy vs Backdoor Accuracy")
plt.legend()
plt.grid(True)

# 保存图片
plt.savefig(img_file, dpi=300, bbox_inches="tight")
print(f"图像已保存为 {img_file}")

# 显示图（如果有 GUI）
try:
    plt.show()
except Exception:
    print("当前环境不支持直接显示图像，请查看保存的图片。")
