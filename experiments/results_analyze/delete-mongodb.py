from pymongo import MongoClient
from bson import ObjectId

# 连接到 MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['backdoor']

# 删除集合中的所有文档
collection = db['5x5:evil']
result = collection.delete_many({})  # 空过滤器表示删除所有文档
print(f"已删除 {result.deleted_count} 个文档")

#删除某一条数据
#result = collection.delete_one({"_id": ObjectId("68c7dee75b2225c1964e8b69")})

# 删除所有 args.learning_rate 为 0.01 的文档
#result = collection.delete_many({"args.learning_rate": 0.01})
#print(f"已删除 {result.deleted_count} 个文档")

# 检查集合中的文档数量
count = collection.count_documents({})
print(f"集合中剩余文档数量: {count}")
