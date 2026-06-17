import json

# 读取 JSON 文件
json_path = r'F:\Study\Cross-Modal-Retrieval\GOAL\datasets\DCI_train_del_org.json'
replace_src = '../DCI/dataset/densely_captioned_images/dataset/scripts/densely_captioned_images/photos/'
replace_dst = 'F:\\Study\\Cross-Modal-Retrieval\\DCI\\data\\densely_captioned_images\\photos\\'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    if 'filename' in item and isinstance(item['filename'], str):
        item['filename'] = item['filename'].replace(replace_src, replace_dst)

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
