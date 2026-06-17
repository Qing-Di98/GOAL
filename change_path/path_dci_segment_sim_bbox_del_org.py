# 直接替换文件内容中的路径字符串
json_path = r'F:\Study\Cross-Modal-Retrieval\GOAL\datasets\DCI_segment_sim_bbox_del_org.json'
old = '/data1/dataset/wh/DCI/segment_with_background_DCI_train_set_max_0.01/'
new = r'F:\\Study\\Cross-Modal-Retrieval\\GOAL_data\\DCI\\segment_with_background_DCI_train_set_max_0.01\\'
with open(json_path, 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace(old, new)
with open(json_path, 'w', encoding='utf-8') as f:
    f.write(text)
print('done')
