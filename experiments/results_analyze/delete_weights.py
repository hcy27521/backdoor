import os
from pymongo import MongoClient

def delete_failed_weights():
    # 连接本地 MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    
    # 选择数据库和集合
    prefix = "tm2_v3:handcrafted"
    db = client["backdoor"]
    col = db[prefix]
    
    # 权重文件的基础目录
    base_dir = "/home/wyl/backdoor/experiments"
    
    # 查询所有success为FALSE的文档
    failed_docs = col.find({"success": "FALSE"})
    
    deleted_count = 0
    not_found_count = 0
    
    for doc in failed_docs:
        # 从result字段中提取权重文件路径
        result = doc.get("result", {})
        
        # 查找权重文件路径 - 键以"weights/"开头且以".pth"结尾
        weight_path = None
        for key in result:
            if key.startswith("weights/gtsb_handcrafted") and key.endswith(".pth"):
                weight_path = key
                break
        
        # 如果找到权重文件路径，转换为绝对路径并尝试删除文件
        if weight_path:
            # 将相对路径转换为绝对路径
            absolute_path = os.path.join(base_dir, weight_path)
            print(f"找到权重文件: {absolute_path}")
            
            if os.path.exists(absolute_path):
                try:
                    os.remove(absolute_path)
                    print(f"已删除: {absolute_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除失败 {absolute_path}: {e}")
            else:
                print(f"文件不存在: {absolute_path}")
                not_found_count += 1
        else:
            print("未找到权重文件路径")
    
    print(f"\n删除完成: 成功删除 {deleted_count} 个文件, {not_found_count} 个文件未找到")

if __name__ == "__main__":
    delete_failed_weights()