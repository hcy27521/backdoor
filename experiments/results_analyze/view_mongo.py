
from pymongo import MongoClient
import pandas as pd

# 连接本地 MongoDB
client = MongoClient("mongodb://localhost:27017/")

# 选择数据库和集合
prefix = "tm2_v3_btsc_badnet:finetune"
db = client["backdoor"]
col = db[prefix]

# 查询所有数据
cursor = col.find()

# 转换为 DataFrame（方便看表格）
df = pd.DataFrame(list(cursor))

# 显示前 10 条记录
print("前 10 条记录：")
print(df.head(10))

# 导出为 CSV
df.to_csv(f"{prefix}.csv", index=False)
print("\n✅ 数据已导出")

