"""将数据集JSON中的Windows路径替换为Linux集群路径"""
import json
import os
import sys

# 路径映射配置
PATH_MAPPINGS = [
    {
        "old": r"F:\Study\Cross-Modal-Retrieval\GOAL_data\DCI\segment_with_background_DCI_train_set_max_0.01\\",
        "new": "/home/scc/pb24111693/GOAL_data/DCI/segment_with_background_DCI_train_set_max_0.01/"
    },
    {
        "old": r"F:\Study\Cross-Modal-Retrieval\GOAL_data\docci\\",
        "new": "/home/scc/pb24111693/GOAL_data/docci/"
    },
    {
        "old": r"F:\Study\Cross-Modal-Retrieval\GOAL_data\\",
        "new": "/home/scc/pb24111693/GOAL_data/"
    },
]

def convert_paths(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        text = f.read()

    for mapping in PATH_MAPPINGS:
        text = text.replace(mapping["old"], mapping["new"])

    with open(json_file, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Converted paths in: {json_file}")

if __name__ == "__main__":
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
    for f in os.listdir(datasets_dir):
        if f.endswith('.json'):
            convert_paths(os.path.join(datasets_dir, f))
    print("All done!")
