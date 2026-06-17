"""Fix Windows paths to Linux in dataset JSON files"""
import json, os

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")

for fname in os.listdir(DATASETS_DIR):
    if not fname.endswith('.json'):
        continue
    path = os.path.join(DATASETS_DIR, fname)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    changed = False
    for item in data:
        old_fn = item.get('original_filename', '')
        if 'F:' in old_fn or '\\\\' in old_fn:
            item['original_filename'] = old_fn.replace('\\', '/').replace(
                'F:/Study/Cross-Modal-Retrieval/GOAL_data',
                '/home/scc/pb24111693/GOAL_data')
            changed = True
        for seg in item.get('segment', []):
            old_seg = seg.get('filename', '')
            if 'F:' in old_seg or '\\\\' in old_seg:
                seg['filename'] = old_seg.replace('\\', '/').replace(
                    'F:/Study/Cross-Modal-Retrieval/GOAL_data',
                    '/home/scc/pb24111693/GOAL_data')
                changed = True

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f'{fname}: converted')
    else:
        print(f'{fname}: no change needed')

print('Done!')
