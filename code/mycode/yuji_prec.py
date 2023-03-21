import numpy as np

# 多分类score
# predictions_file='/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/zhuguanqi/plm_gnn/ckpt/mt_intent_v2/bert_2023-01-17_20:41:29/test_scores.npy'
# SimCSE score
# predictions_file = '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/zhuguanqi/plm_gnn/ckpt/mt_intent_v2/node_bertgnn_2023-01-18_14:37:10/test_scores.npy'
# predictions_file = '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/zhuguanqi/plm_gnn/ckpt/mt_intent_v2/node_cls2_l1_2023-01-17_23:11:18/test_scores.npy'
# predictions_file = '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/zhuguanqi/plm_gnn/ckpt/mt_intent_v2/node_bertgnn_2023-01-19_11:08:03/test_scores.npy'
# predictions_file = '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/zhuguanqi/plm_gnn/ckpt/mt_intent_v2/node_bertgnn_2023-01-19_11:00:53/test_scores.npy'
# predictions_file = 'ckpt/mt_intent_v2/node_bertgnn_2023-01-20_23:41:29/test_scores.npy'
predictions_file = 'ckpt/mt_intent_v2/node_bertgnn_2023-01-23_10:02:48/test_scores.npy'

predictions = np.argmax(np.load(predictions_file), axis=1)
print(predictions.shape)

label_file = '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/zhuguanqi/plm_gnn/data/intent_v2/labels.txt'
labelid_name_dict = {}
cnt = 0
with open(label_file,'r') as f_test:
    for line in f_test:
        labelid_name_dict[cnt] = line.strip().split('@')
        cnt += 1
test_file = '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/zhuguanqi/plm_gnn/data/intent_v2/test.txt'

test_query_labels = {}
test_query_predicts = {}

cnt = 0
with open(test_file,'r') as f_test:
    for line in f_test:
        splits = line.strip().split('\t')
        query = splits[0]
        label = int(splits[1])
        if query not in test_query_labels:
            test_query_labels[query] = set()
        test_query_labels[query].add(label)
        test_query_predicts[query] = int(predictions[cnt])
        cnt += 1

print(f"test length={len(test_query_predicts)}")
hit = 0
miss = 0
for key,labels in test_query_labels.items():
    if test_query_predicts[key] in labels:
        hit += 1
    else:
        miss += 1
print(f"hit={hit}, miss={miss}, precision_top1={hit*1.0/(hit+miss)}")

# (35464,)
# test length=11626
# hit=9140, miss=2486, precision_top1=0.7861689317047996