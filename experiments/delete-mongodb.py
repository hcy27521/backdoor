from pymongo import MongoClient

# 连接到 MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['backdoor']

# 删除集合中的所有文档
collection = db['tm1:cifar:handcrafted:v2']
result = collection.delete_many({})  # 空过滤器表示删除所有文档
print(f"已删除 {result.deleted_count} 个文档")

# 检查集合中的文档数量
count = collection.count_documents({})
print(f"集合中剩余文档数量: {count}")
