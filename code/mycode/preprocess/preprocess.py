from pathlib import Path
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import sys
sys.path.append("../../..")
sys.path.append("../..")
sys.path.append(".")
from module import load_token_mapping
root_dir = Path('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/zhuguanqi/plm_gnn')

# def load_token_mapping(path):
#     idx2token, token2idx = [], {}
#     with open(path, 'r') as f:
#         for i, line in enumerate(f):
#             line = line.strip()
#             idx2token.append(line)
#             token2idx[line] = i
#     return idx2token, token2idx

def save_quert_and_sup():
    queries, sups = set(), set()
    with open(root_dir / 'data/mt_new/new_train', 'r') as f:
        for line in f:
            q, s, _ = line.strip().split('\t')
            queries.add(q.strip())
            sups.add(s.strip())
    with open(root_dir / 'data/mt_new/new_val', 'r') as f:
        for line in f:
            q, s, _ = line.strip().split('\t')
            queries.add(q.strip())
            sups.add(s.strip())
    with open(root_dir / 'data/mt_new/new_test', 'r') as f:
        for line in f:
            q, s, _ = line.strip().split('\t')
            queries.add(q.strip())
            sups.add(s.strip())
    with open(root_dir / 'data/mt_new/queries', 'w') as f:
        for q in queries:
            f.write(f'{q}\n')
    with open(root_dir / 'data/mt_new/sups', 'w') as f:
        for s in sups:
            f.write(f'{s}\n')

def tokenize():
    tokenizer = AutoTokenizer.from_pretrained('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yangyang113/distillation_clean/plms/smallbert_yuji/auto-medium/')
    queries, _ = load_token_mapping(root_dir / 'data/mt_new/queries')
    spus, _ = load_token_mapping(root_dir / 'data/mt_new/spus')
    query_encode = tokenizer(queries, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    spu_encode = tokenizer(spus, padding='max_length', truncation=True, max_length=32,return_tensors='pt')
    print(query_encode['input_ids'].shape, spu_encode['input_ids'].shape)
    for k, v in query_encode.items():
        torch.save(v, root_dir / f'data/mt_new/q_{k}.pt')
    for k, v in spu_encode.items():
        torch.save(v, root_dir / f'data/mt_new/s_{k}.pt')

def isOneEditDistance(s: str, t: str) -> bool:
    distance = len(s) - len(t)
    if abs(distance) > 1:
        return False
    if not s or not t:
        return s != t
    
    edit = 0
    i,j = 0,0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
            j += 1
        else:
            if edit:
                return False

            if distance == 1: # 删除
                i += 1
            elif distance == -1:  # 插入
                j += 1
            else:   # 替换
                i += 1
                j += 1 
            edit += 1
    if i < len(s):
        return edit == 0
    if j < len(t):
        return edit == 0
    return i == len(s) and j == len(t) and edit == 1
def buid_one_ed_graph():
    i2q, q2i  = load_token_mapping('queries')
    with open('one_edit.graph', 'w') as f:
        for i in tqdm(range(len(i2q))):
            for j in range(i + 1, len(i2q)):
                if isOneEditDistance(i2q[i], i2q[j]):
                    f.write(f'{i2q[i]}\t{i2q[j]}\n')
tokenize()
# save_quert_and_sup()
# buid_one_ed_graph()