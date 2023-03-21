from ast import Tuple
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import json
import torch
from dataclasses import dataclass
from transformers import BertTokenizerFast, AutoTokenizer
from sentence_transformers import InputExample
from pathlib import Path
import random
import dgl
from dataclasses import field
from dgl.dataloading import BlockSampler
try:
    from utils import load_token_mapping, load_item_mapping
except ImportError:
    from .utils import load_token_mapping, load_item_mapping

####################intent classification##########################
class IntentData(Dataset):
    def __init__(self, data_path):
        self.query, self.label, self.label_texts = [], [], []
        with open(data_path, 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                data = line.strip().split('\t')
                query, label, label_t = data[0], data[1], data[-1]
                self.query.append(query.strip())
                self.label.append(int(label))
                self.label_texts.append(label_t)

    def __getitem__(self, index):
        return self.query[index], self.label[index], self.label_texts[index]

    def __len__(self):
        return len(self.label)


@dataclass
class DataCollatorForIntentClassification:
    token_length: int = 32
    tokenizer: Union[BertTokenizerFast, str] = "bert-base-uncased"

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_queries, batch_labels = [], []
        for sample in samples:
            query, label, label_t = sample
            batch_queries.append(query)
            # batch_label_texts.append(label_t)
            batch_labels.append(label)
        encoded_dict = self.tokenizer(batch_queries, padding=True, truncation=True, max_length=self.token_length, return_tensors='pt')
        encoded_dict['labels'] = torch.tensor(batch_labels, dtype=torch.long)
        return encoded_dict


class DataCollatorForICYJ:
    def __init__(self, query_path):
        _, q2i = load_token_mapping(query_path)
        self.q2i = q2i
    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_queries, batch_labels = [], []
        ret = {}
        for sample in samples:
            query, label, label_t = sample
            query = query.strip()
            if query in self.q2i:
                qi = self.q2i[query]
            else:
                qi = len(self.q2i)
            batch_queries.append(qi)
            # batch_label_texts.append(label_t)
            batch_labels.append(label)
        ret['sentence_features'] = torch.tensor(batch_queries, dtype=torch.long)
        ret['labels'] = torch.tensor(batch_labels, dtype=torch.long)
        return ret

@dataclass
class DataCollatorForIntent2Tower:
    token_length: int = 32
    tokenizer: Union[BertTokenizerFast, str] = "bert-base-uncased"
    label_path: str = 'data/intent/labels.txt'

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer)
        self.label_texts = []
        with open(self.label_path, 'r') as f:
            for line in f:
                _, t = line.strip().split('\t')
                self.label_texts.append(t)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_queries, batch_pos = [], []
        for sample in samples:
            query, _, label_t = sample
            batch_queries.append(query)
            # batch_label_texts.append(label_t)
            batch_pos.append(label_t)
        batch_neg = [random.randint(0, len(self.label_texts) - 1) for _ in range(len(batch_pos))]
        batch_neg = [self.label_texts[i] for i in batch_neg]
        # encoded_dict = self.tokenizer(batch_queries, padding=True, truncation=True, max_length=self.token_length, return_tensors='pt')
        sentence_features = []
        ret = {}
        tokenized = self.tokenizer(batch_queries, padding=True, truncation=True, max_length=self.token_length, return_tensors='pt')
        sentence_features.append(tokenized)
        tokenized = self.tokenizer(batch_pos, padding=True, truncation=True, max_length=self.token_length, return_tensors='pt')
        sentence_features.append(tokenized)
        tokenized = self.tokenizer(batch_neg, padding=True, truncation=True, max_length=self.token_length, return_tensors='pt')
        sentence_features.append(tokenized)
        ret['sentence_features'] = sentence_features
        return ret

class DC4ICGNN:
    def __init__(self, query_path, graph_path, fanouts):
        _, self.q2i = load_token_mapping(query_path)
        g, _ = dgl.load_graphs(graph_path)
        self.g = g[0]
        self.sampler = dgl.dataloading.NeighborSampler(fanouts=fanouts)
    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_queries, batch_labels = [], []
        ret = {}
        for sample in samples:
            query, label, label_t = sample
            qi = self.q2i[query]
            batch_queries.append(qi)
            # batch_label_texts.append(label_t)
            batch_labels.append(label)
        input_nodes, output_nodes, blocks = self.sampler.sample(self.g, {'query': list(set(batch_queries))})
        output_nodes_map = {nid: i for i, nid in enumerate(output_nodes['query'])}
        output_nodes = [output_nodes_map[nid] for nid in batch_queries]
        ret['input_nodes'] = input_nodes
        ret['output_nodes'] = torch.tensor(output_nodes, dtype=torch.long)
        ret['blocks'] = blocks
        ret['labels'] = torch.tensor(batch_labels, dtype=torch.long)
        return ret


class DC4ICBERTGNN:
    def __init__(self, query_path, dataset_path, graph_path, fanouts):
        _, self.q2i = load_token_mapping(query_path)
        g, _ = dgl.load_graphs(graph_path)
        self.g = g[0]
        self.sampler = dgl.dataloading.NeighborSampler(fanouts=fanouts)
        self.query = {}
        dataset_path = Path(dataset_path)
        self.query['input_ids'] = torch.load(dataset_path / 'query_input_ids.pt')
        self.query['attention_mask'] = torch.load(dataset_path / 'query_attention_mask.pt')
        self.query['token_type_ids'] = torch.load(dataset_path / 'query_token_type_ids.pt')
        # self.query_num = query['input_ids'].shape[0]
        self.item = {}
        self.item['input_ids'] = torch.load(dataset_path / 'item_input_ids.pt')
        self.item['attention_mask'] = torch.load(dataset_path / 'item_attention_mask.pt')
        self.item['token_type_ids'] = torch.load(dataset_path / 'item_token_type_ids.pt')

        self.cate = {}
        self.cate['input_ids'] = torch.load(dataset_path / 'cate_input_ids.pt')
        self.cate['attention_mask'] = torch.load(dataset_path / 'cate_attention_mask.pt')
        self.cate['token_type_ids'] = torch.load(dataset_path / 'cate_token_type_ids.pt')


    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_queries, batch_labels = [], []
        ret = {}
        for sample in samples:
            query, label, label_t = sample
            qi = self.q2i[query]
            batch_queries.append(qi)
            # batch_label_texts.append(label_t)
            batch_labels.append(label)
        input_nodes, output_nodes, blocks = self.sampler.sample(self.g, {'query': list(set(batch_queries))})
        sentence_features = []
        tokenized = {}
        tokenized['input_ids'] = self.query['input_ids'][input_nodes['query']]
        tokenized['attention_mask'] = self.query['attention_mask'][input_nodes['query']]
        tokenized['token_type_ids'] = self.query['token_type_ids'][input_nodes['query']]
        sentence_features.append(tokenized)
        tokenized = {}
        tokenized['input_ids'] = self.item['input_ids'][input_nodes['item']]
        tokenized['attention_mask'] = self.item['attention_mask'][input_nodes['item']]
        tokenized['token_type_ids'] = self.item['token_type_ids'][input_nodes['item']]
        sentence_features.append(tokenized)
        tokenized = {}
        tokenized['input_ids'] = self.cate['input_ids'][input_nodes['cate']]
        tokenized['attention_mask'] = self.cate['attention_mask'][input_nodes['cate']]
        tokenized['token_type_ids'] = self.cate['token_type_ids'][input_nodes['cate']]
        sentence_features.append(tokenized)
        output_nodes_map = {nid: i for i, nid in enumerate(output_nodes['query'])}
        output_nodes = [output_nodes_map[nid] for nid in batch_queries]
        ret['sentence_features'] = sentence_features
        ret['output_nodes'] = torch.tensor(output_nodes, dtype=torch.long)
        ret['blocks'] = blocks
        ret['labels'] = torch.tensor(batch_labels, dtype=torch.long)
        return ret

###################################################################
class BertData(Dataset):
    def __init__(self, data_path, aug=False, aug_path=None):
        self.query, self.sup, self.label = [], [], []
        with open(data_path, 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                data = line.strip().split('\t')
                query, sup, label = data[0], data[1], data[-1]
                self.query.append(query.strip())
                self.sup.append(sup.strip())
                self.label.append(int(label))
        
        if aug:
            with open(aug_path, 'r') as f:
                for line in f:
                    if line == '\n':
                        continue
                    arr = line.strip().split('\t')
                    query = arr[0]
                    sup = arr[1]
                    label = arr[-1]
                    self.query.append(query.strip())
                    self.sup.append(sup.strip())
                    self.label.append(int(label))

    def __getitem__(self, index):
        return self.query[index], self.sup[index], self.label[index]

    def __len__(self):
        return len(self.label)

class SBertDataDistill(Dataset):
    def __init__(self, data_path, soft_label_path):
        self.query, self.sup, self.hard_label = [], [], []
        with open(data_path, 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                data = line.strip().split('\t')
                query, sup, label = data[0], data[1], data[-1]
                self.query.append(query.strip())
                self.sup.append(sup.strip())
                self.hard_label.append(int(label))
        self.soft_label = None
        if soft_label_path:
            self.soft_label = np.load(soft_label_path)

    def __getitem__(self, index):
        if self.soft_label is not None:
            return self.query[index], self.sup[index], self.hard_label[index], self.soft_label[index]
        else:
            return self.query[index], self.sup[index], self.hard_label[index], None

    def __len__(self):
        return len(self.hard_label)

class SBertDataDistillQ2q(Dataset):
    def __init__(self, query_path, sup_path, data_path, soft_label_path):
        _, q2i = load_token_mapping(query_path)
        _, s2i = load_token_mapping(sup_path)
        
        self.query, self.sup, self.hard_label = [], [], []
        with open(data_path, 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                data = line.strip().split('\t')
                query, sup, label = data[0], data[1], data[-1]
                self.query.append(q2i[query.strip()])
                self.sup.append(s2i[sup.strip()])
                self.hard_label.append(int(label))

        self.soft_label = None
        if soft_label_path:
            self.soft_label = np.load(soft_label_path)

    def __getitem__(self, index):
        if self.soft_label is not None:
            return self.query[index], self.sup[index], self.hard_label[index], self.soft_label[index]
        else:
            return self.query[index], self.sup[index], self.hard_label[index], None
    def __len__(self):
        return len(self.hard_label)


@dataclass
class DataCollatorForTuning:
    token_length: int = 32
    tokenizer: Union[BertTokenizerFast, str] = "bert-base-uncased"

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_queries, batch_sups, batch_labels = [], [], []
        for sample in samples:
            query, sup, label = sample
            batch_queries.append(query)
            batch_sups.append(sup)
            batch_labels.append(label)
        encoded_dict = self.tokenizer(batch_queries, batch_sups, padding=True, truncation=True, max_length=self.token_length, return_tensors='pt')
        encoded_dict['labels'] = torch.tensor(batch_labels, dtype=torch.long)
        return encoded_dict


@dataclass
class DataCollatorForPretraining:
    token_length: int = 32
    tokenizer: Union[BertTokenizerFast, str] = "bert-base-uncased"
    mlm_probability: float = 0.15

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_queries, batch_sups, batch_labels = [], [], []
        for sample in samples:
            query, sup, label = sample
            batch_queries.append(query)
            batch_sups.append(sup)
            # 对于nsp任务，0代表连续，1代表不连续，所以要反一下
            batch_labels.append(0 if label == 1 else 1)
        batch = self.tokenizer(batch_queries, batch_sups, padding=True, truncation=True, max_length=self.token_length, return_tensors='pt', return_special_tokens_mask=True)
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        batch['next_sentence_label'] = torch.tensor(batch_labels, dtype=torch.long)
        return batch

    def torch_mask_tokens(self, inputs, special_tokens_mask = None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class DataCollatorForSBert:
    token_length: int = 32
    tokenizer: Union[AutoTokenizer, str] = "bert-base-uncased"

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_labels = []
        texts = [[] for _ in range(2)]
        ret = {}
        for sample in samples:
            query, sup, label = sample
            texts[0].append(query)
            texts[1].append(sup)
            batch_labels.append(label)
        sentence_features = []
        for idx in range(2):
            tokenized = self.tokenizer(texts[idx], padding='max_length', truncation=True, max_length=self.token_length, return_tensors='pt')
            sentence_features.append(tokenized)
        ret['sentence_features'] = sentence_features
        ret['labels'] = torch.tensor(batch_labels, dtype=torch.long)
        return ret


@dataclass
class DataCollatorForSBertDistill:
    token_length: int = 32
    tokenizer: Union[AutoTokenizer, str] = "bert-base-uncased"

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_hard_labels = []
        batch_soft_labels = []
        texts = [[] for _ in range(2)]
        ret = {}
        for sample in samples:
            query, sup, hard_label, soft_label = sample
            texts[0].append(query)
            texts[1].append(sup)
            batch_hard_labels.append(hard_label)
            batch_soft_labels.append(soft_label)
        sentence_features = []
        for idx in range(2):
            tokenized = self.tokenizer(texts[idx], padding='max_length', truncation=True, max_length=self.token_length, return_tensors='pt')
            sentence_features.append(tokenized)
        ret['sentence_features'] = sentence_features
        ret['hard_labels'] = torch.tensor(batch_hard_labels, dtype=torch.long)
        ret['soft_labels'] = torch.tensor(batch_soft_labels, dtype=torch.float)
        return ret

class DataCollatorForSBertDistillQ2q:

    def __init__(self, dataset_path, q2q_g_path, pos_sample=True, neg_sample=True):
        self.pos_sample = pos_sample
        self.neg_sample = neg_sample
        dataset_path = Path(dataset_path)
        self.query, self.sup = {}, {}
        self.query['input_ids'] = torch.load(dataset_path / 'q_input_ids.pt')
        self.query['attention_mask'] = torch.load(dataset_path / 'q_attention_mask.pt')
        self.query['token_type_ids'] = torch.load(dataset_path / 'q_token_type_ids.pt')

        self.sup['input_ids'] = torch.load(dataset_path / 's_input_ids.pt')
        self.sup['attention_mask'] = torch.load(dataset_path / 's_attention_mask.pt')
        self.sup['token_type_ids'] = torch.load(dataset_path / 's_token_type_ids.pt')
        q2q_g, _ = dgl.load_graphs(q2q_g_path)
        self.q2q_g = q2q_g[0]

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_hard_labels = []
        batch_soft_labels = []
        texts = [[] for _ in range(2)]
        ret = {}
        for sample in samples:
            query, sup, hard_label, soft_label = sample
            texts[0].append(query)
            texts[1].append(sup)
            batch_hard_labels.append(hard_label)
            if soft_label is not None:
                batch_soft_labels.append(soft_label)
        sentence_features = []
        tokenized = {}
        tokenized['input_ids'] = self.query['input_ids'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['attention_mask'] = self.query['attention_mask'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['token_type_ids'] = self.query['token_type_ids'][torch.tensor(texts[0], dtype=torch.long)]
        sentence_features.append(tokenized)
        # TODO tokenized BUG
        tokenized = {}
        tokenized['input_ids'] = self.sup['input_ids'][torch.tensor(texts[1], dtype=torch.long)]
        tokenized['attention_mask'] = self.sup['attention_mask'][torch.tensor(texts[1], dtype=torch.long)]
        tokenized['token_type_ids'] = self.sup['token_type_ids'][torch.tensor(texts[1], dtype=torch.long)]
        sentence_features.append(tokenized)
        ret['sentence_features'] = sentence_features
        ret['hard_labels'] = torch.tensor(batch_hard_labels, dtype=torch.long)
        ret['soft_labels'] = torch.tensor(batch_soft_labels, dtype=torch.float) if batch_soft_labels else None
        query_sentence_features = []
        if self.pos_sample:
            sg = dgl.sampling.sample_neighbors(self.q2q_g, texts[0], 1)
            pos_q = sg.edges()[0]
            tokenized = {}
            tokenized['input_ids'] = self.query['input_ids'][pos_q]
            tokenized['attention_mask'] = self.query['attention_mask'][pos_q]
            tokenized['token_type_ids'] = self.query['token_type_ids'][pos_q]
            query_sentence_features.append(tokenized)
        if self.neg_sample:
            neg_q = torch.randint(0, self.q2q_g.num_nodes(), (len(texts[0]),), dtype=torch.long)
            # TODO tokenized BUG
            tokenized = {}
            tokenized['input_ids'] = self.sup['input_ids'][neg_q]
            tokenized['attention_mask'] = self.sup['attention_mask'][neg_q]
            tokenized['token_type_ids'] = self.sup['token_type_ids'][neg_q]
            query_sentence_features.append(tokenized)
        ret['query_sentence_features'] = query_sentence_features if query_sentence_features else None
        return ret


class DataCollatorForSBertDistillQ2s:

    def __init__(self, dataset_path, q2s_g_path, pos_sample=True, neg_sample=True):
        self.pos_sample = pos_sample
        self.neg_sample = neg_sample
        dataset_path = Path(dataset_path)
        query, spu = {}, {}
        query['input_ids'] = torch.load(dataset_path / 'q_input_ids.pt')
        query['attention_mask'] = torch.load(dataset_path / 'q_attention_mask.pt')
        query['token_type_ids'] = torch.load(dataset_path / 'q_token_type_ids.pt')
        self.query_num = query['input_ids'].shape[0]
        spu['input_ids'] = torch.load(dataset_path / 's_input_ids.pt')
        spu['attention_mask'] = torch.load(dataset_path / 's_attention_mask.pt')
        spu['token_type_ids'] = torch.load(dataset_path / 's_token_type_ids.pt')
        self.node = {}
        self.node['input_ids'] = torch.cat((query['input_ids'], spu['input_ids']), dim=0)
        self.node['attention_mask'] = torch.cat((query['attention_mask'], spu['attention_mask']), dim=0)
        self.node['token_type_ids'] = torch.cat((query['token_type_ids'], spu['token_type_ids']), dim=0)

        q2s_g, _ = dgl.load_graphs(q2s_g_path)
        self.q2s_g = q2s_g[0]

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_hard_labels = []
        batch_soft_labels = []
        texts = [[] for _ in range(2)]
        ret = {}
        for sample in samples:
            query, spu, hard_label, soft_label = sample
            texts[0].append(query)
            texts[1].append(spu + self.query_num)
            batch_hard_labels.append(hard_label)
            if soft_label is not None:
                batch_soft_labels.append(soft_label)
        sentence_features = []
        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['attention_mask'] = self.node['attention_mask'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['token_type_ids'] = self.node['token_type_ids'][torch.tensor(texts[0], dtype=torch.long)]
        sentence_features.append(tokenized)
        # TODO tokenized BUG
        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][torch.tensor(texts[1], dtype=torch.long)]
        tokenized['attention_mask'] = self.node['attention_mask'][torch.tensor(texts[1], dtype=torch.long)]
        tokenized['token_type_ids'] = self.node['token_type_ids'][torch.tensor(texts[1], dtype=torch.long)]
        sentence_features.append(tokenized)
        ret['sentence_features'] = sentence_features
        ret['hard_labels'] = torch.tensor(batch_hard_labels, dtype=torch.long)
        ret['soft_labels'] = torch.tensor(batch_soft_labels, dtype=torch.float) if batch_soft_labels else None
        query_sentence_features = []
        if self.pos_sample:
            sg = dgl.sampling.sample_neighbors(self.q2s_g, texts[0], 1)
            pos_q = sg.edges()[0]
            tokenized = {}
            tokenized['input_ids'] = self.node['input_ids'][pos_q]
            tokenized['attention_mask'] = self.node['attention_mask'][pos_q]
            tokenized['token_type_ids'] = self.node['token_type_ids'][pos_q]
            query_sentence_features.append(tokenized)
        if self.neg_sample:
            # neg_q = torch.randint(0, self.q2s_g.num_nodes(), (len(texts[0]),), dtype=torch.long)
            # tokenized['input_ids'] = self.node['input_ids'][neg_q]
            # tokenized['attention_mask'] = self.node['attention_mask'][neg_q]
            # tokenized['token_type_ids'] = self.node['token_type_ids'][neg_q]
            # query_sentence_features.append(tokenized)
            pass
        ret['query_sentence_features'] = query_sentence_features if query_sentence_features else None
        return ret

class DataCollatorForSBertDistillContrastive:

    def __init__(self, dataset_path, query_neigh_g, spu_neigh_g):
        dataset_path = Path(dataset_path)
        query, spu = {}, {}
        query['input_ids'] = torch.load(dataset_path / 'q_input_ids.pt')
        query['attention_mask'] = torch.load(dataset_path / 'q_attention_mask.pt')
        query['token_type_ids'] = torch.load(dataset_path / 'q_token_type_ids.pt')
        self.query_num = query['input_ids'].shape[0]
        spu['input_ids'] = torch.load(dataset_path / 's_input_ids.pt')
        spu['attention_mask'] = torch.load(dataset_path / 's_attention_mask.pt')
        spu['token_type_ids'] = torch.load(dataset_path / 's_token_type_ids.pt')
        self.node = {}
        self.node['input_ids'] = torch.cat((query['input_ids'], spu['input_ids']), dim=0)
        self.node['attention_mask'] = torch.cat((query['attention_mask'], spu['attention_mask']), dim=0)
        self.node['token_type_ids'] = torch.cat((query['token_type_ids'], spu['token_type_ids']), dim=0)
        self.q_neigh_g = None
        self.s_neigh_g = None
        if query_neigh_g:
            q_neigh_g, _ = dgl.load_graphs(query_neigh_g)
            self.q_neigh_g = q_neigh_g[0]
        if spu_neigh_g:
            self.is_s2s = 's2s' in spu_neigh_g
            s_neigh_g, _ = dgl.load_graphs(spu_neigh_g)
            self.s_neigh_g = s_neigh_g[0]


    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_hard_labels = []
        batch_soft_labels = []
        texts = [[] for _ in range(2)]
        ret = {}
        for sample in samples:
            query, spu, hard_label, soft_label = sample
            texts[0].append(query)
            texts[1].append(spu + self.query_num)
            batch_hard_labels.append(hard_label)
            if soft_label is not None:
                batch_soft_labels.append(soft_label)
        sentence_features = []
        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['attention_mask'] = self.node['attention_mask'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['token_type_ids'] = self.node['token_type_ids'][torch.tensor(texts[0], dtype=torch.long)]
        sentence_features.append(tokenized)
        # TODO tokenized BUG
        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][torch.tensor(texts[1], dtype=torch.long)]
        tokenized['attention_mask'] = self.node['attention_mask'][torch.tensor(texts[1], dtype=torch.long)]
        tokenized['token_type_ids'] = self.node['token_type_ids'][torch.tensor(texts[1], dtype=torch.long)]
        sentence_features.append(tokenized)
        ret['sentence_features'] = sentence_features
        ret['hard_labels'] = torch.tensor(batch_hard_labels, dtype=torch.long)
        ret['soft_labels'] = torch.tensor(batch_soft_labels, dtype=torch.float) if batch_soft_labels else None
        query_neighbor_features = []
        spu_neighbor_features = []
        if self.q_neigh_g:
            sg = dgl.sampling.sample_neighbors(self.q_neigh_g, texts[0], 1)
            neighbors = sg.edges()[0]
            tokenized = {}
            tokenized['input_ids'] = self.node['input_ids'][neighbors]
            tokenized['attention_mask'] = self.node['attention_mask'][neighbors]
            tokenized['token_type_ids'] = self.node['token_type_ids'][neighbors]
            query_neighbor_features.append(tokenized)
        if self.s_neigh_g:
            spus = texts[1] if not self.is_s2s else [s - self.query_num for s in texts[1]]
            sg = dgl.sampling.sample_neighbors(self.s_neigh_g, spus, 1)
            neighbors = sg.edges()[0]
            neighbors = neighbors if not self.is_s2s else neighbors + self.query_num
            tokenized = {}
            tokenized['input_ids'] = self.node['input_ids'][neighbors]
            tokenized['attention_mask'] = self.node['attention_mask'][neighbors]
            tokenized['token_type_ids'] = self.node['token_type_ids'][neighbors]
            spu_neighbor_features.append(tokenized)
        ret['query_neighbor_features'] = query_neighbor_features if query_neighbor_features else None
        ret['spu_neighbor_features'] = spu_neighbor_features if spu_neighbor_features else None
        return ret

class DataCollatorForSimcse:

    def __init__(self, dataset_path, g_path, pos_sample=True, neg_sample=True):
        self.pos_sample = pos_sample
        self.neg_sample = neg_sample
        dataset_path = Path(dataset_path)
        query, spu = {}, {}
        query['input_ids'] = torch.load(dataset_path / 'q_input_ids.pt')
        query['attention_mask'] = torch.load(dataset_path / 'q_attention_mask.pt')
        query['token_type_ids'] = torch.load(dataset_path / 'q_token_type_ids.pt')
        self.query_num = query['input_ids'].shape[0]
        spu['input_ids'] = torch.load(dataset_path / 's_input_ids.pt')
        spu['attention_mask'] = torch.load(dataset_path / 's_attention_mask.pt')
        spu['token_type_ids'] = torch.load(dataset_path / 's_token_type_ids.pt')
        self.node = {}
        self.node['input_ids'] = torch.cat((query['input_ids'], spu['input_ids']), dim=0)
        self.node['attention_mask'] = torch.cat((query['attention_mask'], spu['attention_mask']), dim=0)
        self.node['token_type_ids'] = torch.cat((query['token_type_ids'], spu['token_type_ids']), dim=0)

        graph, _ = dgl.load_graphs(g_path)
        graph = graph[0]
        neg_g = dgl.to_homogeneous(graph['neg'])
        self.neg_g = dgl.add_self_loop(neg_g)
        pos_g = dgl.to_homogeneous(graph['pos'])
        self.pos_g = dgl.add_self_loop(pos_g)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        texts = [[] for _ in range(2)]
        ret = {}
        for sample in samples:
            query, spu, _, _ = sample
            texts[0].append(query)
            texts[1].append(spu + self.query_num)
        sentence_features = []
        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['attention_mask'] = self.node['attention_mask'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['token_type_ids'] = self.node['token_type_ids'][torch.tensor(texts[0], dtype=torch.long)]
        sentence_features.append(tokenized)

        if self.pos_sample:
            sg = dgl.sampling.sample_neighbors(self.pos_g, texts[0], 1, edge_dir='out')
            pos_q = sg.edges()[1]
            tokenized = {}
            tokenized['input_ids'] = self.node['input_ids'][pos_q]
            tokenized['attention_mask'] = self.node['attention_mask'][pos_q]
            tokenized['token_type_ids'] = self.node['token_type_ids'][pos_q]
            sentence_features.append(tokenized)
        if self.neg_sample:
            sg = dgl.sampling.sample_neighbors(self.neg_g, texts[0], 1, edge_dir='out')
            neg_src, neg_dst = sg.edges()
            # 如果sample到自身，那么负样本随机采样
            mask = neg_src == neg_dst
            neg_dst[mask] = torch.randint(low=self.query_num, high=self.neg_g.num_nodes(), size=neg_dst[mask].shape)
            tokenized = {}
            tokenized['input_ids'] = self.node['input_ids'][neg_dst]
            tokenized['attention_mask'] = self.node['attention_mask'][neg_dst]
            tokenized['token_type_ids'] = self.node['token_type_ids'][neg_dst]
            sentence_features.append(tokenized)

        ret['sentence_features'] = sentence_features

        return ret

class DataCollatorForSBertPNG:
    def __init__(self, dataset_path, g_path, fanouts, query_path: str = 'data/mt_new/queries', spu_path: str = 'data/mt_new/spus', pos_sample=True, neg_sample=True):
        self.pos_sample = pos_sample
        self.neg_sample = neg_sample
        dataset_path = Path(dataset_path)
        i2q, self.q2i  = load_token_mapping(query_path)
        i2s, s2i = load_token_mapping(spu_path)
        self.s2i = {k: v + len(self.q2i) for k, v in s2i.items()}
        query, spu = {}, {}
        query['input_ids'] = torch.load(dataset_path / 'q_input_ids.pt')
        query['attention_mask'] = torch.load(dataset_path / 'q_attention_mask.pt')
        query['token_type_ids'] = torch.load(dataset_path / 'q_token_type_ids.pt')
        self.query_num = query['input_ids'].shape[0]
        spu['input_ids'] = torch.load(dataset_path / 's_input_ids.pt')
        spu['attention_mask'] = torch.load(dataset_path / 's_attention_mask.pt')
        spu['token_type_ids'] = torch.load(dataset_path / 's_token_type_ids.pt')
        self.node = {}
        self.node['input_ids'] = torch.cat((query['input_ids'], spu['input_ids']), dim=0)
        self.node['attention_mask'] = torch.cat((query['attention_mask'], spu['attention_mask']), dim=0)
        self.node['token_type_ids'] = torch.cat((query['token_type_ids'], spu['token_type_ids']), dim=0)

        graph, _ = dgl.load_graphs(g_path)
        graph = graph[0]
        neg_g = dgl.to_homogeneous(graph['neg'])
        neg_g = dgl.to_bidirected(neg_g)
        self.neg_g = dgl.add_self_loop(neg_g)
        pos_g = dgl.to_homogeneous(graph['pos'])
        pos_g = dgl.to_bidirected(pos_g)
        self.pos_g = dgl.add_self_loop(pos_g)
        self.sampler = NeighborSampler(fanouts=fanouts)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        texts = [[] for _ in range(2)]
        ret = {}
        labels = []
        for sample in samples:
            query, spu, hard_label, _ = sample
            texts[0].append(self.q2i[query])
            texts[1].append(self.s2i[spu])
            labels.append(hard_label)
        graph_blocks = []
        sentence_features = []
        tokenized = {}
        # tokenized['input_ids'] = self.node['input_ids'][torch.tensor(texts[0], dtype=torch.long)]
        # tokenized['attention_mask'] = self.node['attention_mask'][torch.tensor(texts[0], dtype=torch.long)]
        # tokenized['token_type_ids'] = self.node['token_type_ids'][torch.tensor(texts[0], dtype=torch.long)]
        # sentence_features.append(tokenized)
        seed_nodes = texts[0] + texts[1]
        seed_nodes_set = list(set(seed_nodes))
        input_nodes, output_nodes, blocks = self.sampler.sample(self.pos_g, seed_nodes_set)
        output_nodes_map = {nid: i for i, nid in enumerate(output_nodes)}
        output_nodes = [output_nodes_map[nid] for nid in seed_nodes]
        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][input_nodes]
        tokenized['attention_mask'] = self.node['attention_mask'][input_nodes]
        tokenized['token_type_ids'] = self.node['token_type_ids'][input_nodes]
        sentence_features.append(tokenized)
        graph_blocks.append(blocks)

        input_nodes, _, blocks = self.sampler.sample(self.neg_g, seed_nodes_set, is_neg=True)
        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][input_nodes]
        tokenized['attention_mask'] = self.node['attention_mask'][input_nodes]
        tokenized['token_type_ids'] = self.node['token_type_ids'][input_nodes]
        sentence_features.append(tokenized)
        graph_blocks.append(blocks)

        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['attention_mask'] = self.node['attention_mask'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['token_type_ids'] = self.node['token_type_ids'][torch.tensor(texts[0], dtype=torch.long)]
        sentence_features.append(tokenized)
        # TODO tokenized BUG
        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][torch.tensor(texts[1], dtype=torch.long)]
        tokenized['attention_mask'] = self.node['attention_mask'][torch.tensor(texts[1], dtype=torch.long)]
        tokenized['token_type_ids'] = self.node['token_type_ids'][torch.tensor(texts[1], dtype=torch.long)]
        sentence_features.append(tokenized)

        ret['output_nodes'] = torch.tensor(output_nodes, dtype=torch.long)
        ret['sentence_features'] = sentence_features
        ret['graph_blocks'] = graph_blocks
        ret['labels'] = torch.tensor(labels, dtype=torch.long)

        return ret


class DataCollatorForSBertGNN2:
    """
        GNN接在bert之后，串行的结构
    """
    def __init__(self, dataset_path, g_path, fanouts, query_path: str = 'data/mt_new/queries', spu_path: str = 'data/mt_new/spus', is_train=False):
        self.is_train = is_train
        dataset_path = Path(dataset_path)
        i2q, self.q2i  = load_token_mapping(query_path)
        i2s, s2i = load_token_mapping(spu_path)
        self.s2i = {k: v + len(self.q2i) for k, v in s2i.items()}
        query, spu = {}, {}
        query['input_ids'] = torch.load(dataset_path / 'q_input_ids.pt')
        query['attention_mask'] = torch.load(dataset_path / 'q_attention_mask.pt')
        query['token_type_ids'] = torch.load(dataset_path / 'q_token_type_ids.pt')
        self.query_num = query['input_ids'].shape[0]
        spu['input_ids'] = torch.load(dataset_path / 's_input_ids.pt')
        spu['attention_mask'] = torch.load(dataset_path / 's_attention_mask.pt')
        spu['token_type_ids'] = torch.load(dataset_path / 's_token_type_ids.pt')
        self.node = {}
        self.node['input_ids'] = torch.cat((query['input_ids'], spu['input_ids']), dim=0)
        self.node['attention_mask'] = torch.cat((query['attention_mask'], spu['attention_mask']), dim=0)
        self.node['token_type_ids'] = torch.cat((query['token_type_ids'], spu['token_type_ids']), dim=0)

        graph, _ = dgl.load_graphs(g_path)
        graph = graph[0]
        pos_g = dgl.to_homogeneous(graph['pos'])
        pos_g = dgl.to_bidirected(pos_g)
        self.g = dgl.add_self_loop(pos_g)
        self.sampler = dgl.dataloading.NeighborSampler(fanouts=fanouts)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        texts = [[] for _ in range(2)]
        ret = {}
        labels = []
        src, dst = [], []
        for sample in samples:
            query, spu, hard_label, _ = sample
            texts[0].append(self.q2i[query])
            texts[1].append(self.s2i[spu])
            labels.append(hard_label)
            if self.is_train and hard_label == 1:
                src.append(self.q2i[query])
                dst.append(self.s2i[spu])
                src.append(self.s2i[spu])
                dst.append(self.q2i[query])
        sentence_features = []
        tokenized = {}
        # tokenized['input_ids'] = self.node['input_ids'][torch.tensor(texts[0], dtype=torch.long)]
        # tokenized['attention_mask'] = self.node['attention_mask'][torch.tensor(texts[0], dtype=torch.long)]
        # tokenized['token_type_ids'] = self.node['token_type_ids'][torch.tensor(texts[0], dtype=torch.long)]
        # sentence_features.append(tokenized)
        seed_nodes = texts[0] + texts[1]
        seed_nodes_set = list(set(seed_nodes))
        exclude_eids = self.g.edge_ids(src, dst) if self.is_train else None
        input_nodes, output_nodes, blocks = self.sampler.sample(self.g, seed_nodes_set, exclude_eids=exclude_eids)
        output_nodes_map = {nid: i for i, nid in enumerate(output_nodes)}
        output_nodes = [output_nodes_map[nid] for nid in seed_nodes]
        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][input_nodes]
        tokenized['attention_mask'] = self.node['attention_mask'][input_nodes]
        tokenized['token_type_ids'] = self.node['token_type_ids'][input_nodes]
        sentence_features.append(tokenized)

        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['attention_mask'] = self.node['attention_mask'][torch.tensor(texts[0], dtype=torch.long)]
        tokenized['token_type_ids'] = self.node['token_type_ids'][torch.tensor(texts[0], dtype=torch.long)]
        sentence_features.append(tokenized)
        # TODO tokenized BUG
        tokenized = {}
        tokenized['input_ids'] = self.node['input_ids'][torch.tensor(texts[1], dtype=torch.long)]
        tokenized['attention_mask'] = self.node['attention_mask'][torch.tensor(texts[1], dtype=torch.long)]
        tokenized['token_type_ids'] = self.node['token_type_ids'][torch.tensor(texts[1], dtype=torch.long)]
        sentence_features.append(tokenized)

        ret['output_nodes'] = torch.tensor(output_nodes, dtype=torch.long)
        ret['sentence_features'] = sentence_features
        ret['blocks'] = blocks
        ret['labels'] = torch.tensor(labels, dtype=torch.long)

        return ret


class DataCollatorForGNN:
    def __init__(self, dataset_path, g_path, fanouts, query_path: str = 'data/mt_new/queries', spu_path: str = 'data/mt_new/spus', is_train=False):
        self.is_train = is_train
        dataset_path = Path(dataset_path)
        i2q, self.q2i  = load_token_mapping(query_path)
        i2s, s2i = load_token_mapping(spu_path)
        self.s2i = {k: v + len(self.q2i) for k, v in s2i.items()}
        query, spu = {}, {}
        query['input_ids'] = torch.load(dataset_path / 'q_input_ids.pt')
        query['attention_mask'] = torch.load(dataset_path / 'q_attention_mask.pt')
        query['token_type_ids'] = torch.load(dataset_path / 'q_token_type_ids.pt')
        self.query_num = query['input_ids'].shape[0]
        spu['input_ids'] = torch.load(dataset_path / 's_input_ids.pt')
        spu['attention_mask'] = torch.load(dataset_path / 's_attention_mask.pt')
        spu['token_type_ids'] = torch.load(dataset_path / 's_token_type_ids.pt')
        self.node = {}
        self.node['input_ids'] = torch.cat((query['input_ids'], spu['input_ids']), dim=0)
        self.node['attention_mask'] = torch.cat((query['attention_mask'], spu['attention_mask']), dim=0)
        self.node['token_type_ids'] = torch.cat((query['token_type_ids'], spu['token_type_ids']), dim=0)

        graph, _ = dgl.load_graphs(g_path)
        graph = graph[0]
        pos_g = dgl.to_homogeneous(graph['pos'])
        pos_g = dgl.to_bidirected(pos_g)
        self.g = dgl.add_self_loop(pos_g)
        self.sampler = dgl.dataloading.NeighborSampler(fanouts=fanouts)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        texts = [[] for _ in range(2)]
        ret = {}
        labels = []
        src, dst = [], []
        for sample in samples:
            query, spu, hard_label, _ = sample
            texts[0].append(self.q2i[query])
            texts[1].append(self.s2i[spu])
            labels.append(hard_label)
            if self.is_train and hard_label == 1:
                src.append(self.q2i[query])
                dst.append(self.s2i[spu])
                src.append(self.s2i[spu])
                dst.append(self.q2i[query])
        seed_nodes = texts[0] + texts[1]
        seed_nodes_set = list(set(seed_nodes))
        exclude_eids = self.g.edge_ids(src, dst) if self.is_train else None
        input_nodes, output_nodes, blocks = self.sampler.sample(self.g, seed_nodes_set, exclude_eids=exclude_eids)
        output_nodes_map = {nid: i for i, nid in enumerate(output_nodes)}
        output_nodes = [output_nodes_map[nid] for nid in seed_nodes]
        ret['output_nodes'] = torch.tensor(output_nodes, dtype=torch.long)
        ret['blocks'] = blocks
        ret['labels'] = torch.tensor(labels, dtype=torch.long)

        return ret


@dataclass
class DataCollatorForSBertGNN:
    token_length: int = 32
    tokenizer: Union[AutoTokenizer, str] = "bert-base-uncased"
    query_path: str = 'data/mt_new/queries'
    spu_path: str = 'data/mt_new/spus'
    graph_path: str = 'data/mt_new/q_s_train.graph'
    fanouts: list = field(default_factory=list)
    is_train: bool = True

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        _, self.q2i  = load_token_mapping(self.query_path)
        _, s2i = load_token_mapping(self.spu_path)
        self.s2i = {k: v + len(self.q2i) for k, v in s2i.items()}
        g, _ = dgl.load_graphs(self.graph_path)
        self.g = g[0]
        self.sampler = dgl.dataloading.NeighborSampler(fanouts=self.fanouts)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_hard_labels = []
        batch_soft_labels = []
        texts = [[] for _ in range(2)]
        ret = {}
        query_nodes, spu_nodes = [], []
        src, dst = [], []
        for sample in samples:
            query, spu, hard_label, soft_label = sample
            texts[0].append(query)
            texts[1].append(spu)
            batch_hard_labels.append(hard_label)
            if soft_label is not None:
                batch_soft_labels.append(soft_label)
            query_nodes.append(self.q2i[query])
            spu_nodes.append(self.s2i[spu])
            if self.is_train and hard_label == 1:
                src.append(self.q2i[query])
                dst.append(self.s2i[spu])
                src.append(self.s2i[spu])
                dst.append(self.q2i[query])
        seed_nodes = query_nodes + spu_nodes
        exclude_eids = self.g.edge_ids(src, dst) if self.is_train else None
        _, output_nodes, blocks = self.sampler.sample(self.g, list(set(seed_nodes)), exclude_eids=exclude_eids)
        output_nodes_map = {nid: i for i, nid in enumerate(output_nodes)}
        output_nodes = [output_nodes_map[nid] for nid in seed_nodes]
        sentence_features = []
        for idx in range(2):
            tokenized = self.tokenizer(texts[idx], padding=True, truncation=True, max_length=self.token_length, return_tensors='pt')
            sentence_features.append(tokenized)
        ret['sentence_features'] = sentence_features
        ret['hard_labels'] = torch.tensor(batch_hard_labels, dtype=torch.long)
        ret['soft_labels'] = torch.tensor(batch_soft_labels, dtype=torch.float) if batch_soft_labels else None
        ret['blocks'] = blocks
        ret['output_nodes'] = torch.tensor(output_nodes, dtype=torch.long)
        return ret

@dataclass
class DataCollatorForSBertDGI:
    token_length: int = 32
    tokenizer: Union[AutoTokenizer, str] = "bert-base-uncased"
    query_path: str = 'data/mt_new/queries'
    spu_path: str = 'data/mt_new/spus'
    graph_path: str = 'data/mt_new/q_s_train.graph'
    fanouts: list = field(default_factory=list)
    is_train: bool = True

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        i2q, self.q2i  = load_token_mapping(self.query_path)
        i2s, s2i = load_token_mapping(self.spu_path)
        self.s2i = {k: v + len(self.q2i) for k, v in s2i.items()}
        self.i2node = i2q + i2s
        g, _ = dgl.load_graphs(self.graph_path)
        self.g = g[0]
        self.sampler = dgl.dataloading.NeighborSampler(fanouts=self.fanouts)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_hard_labels = []
        texts = [[] for _ in range(2)]
        ret = {}
        query_nodes, spu_nodes = [], []
        src, dst = [], []
        for sample in samples:
            query, spu, hard_label = sample
            texts[0].append(query)
            texts[1].append(spu)
            batch_hard_labels.append(hard_label)
            query_nodes.append(self.q2i[query])
            spu_nodes.append(self.s2i[spu])
            if self.is_train and hard_label == 1:
                src.append(self.q2i[query])
                dst.append(self.s2i[spu])
                src.append(self.s2i[spu])
                dst.append(self.q2i[query])
        seed_nodes = query_nodes + spu_nodes
        exclude_eids = None
        # exclude_eids = self.g.edge_ids(src, dst) if self.is_train else None
        input_nodes, output_nodes, blocks = self.sampler.sample(self.g, list(set(seed_nodes)), exclude_eids=exclude_eids)
        input_text = [self.i2node[i] for i in input_nodes]
        output_nodes_map = {nid: i for i, nid in enumerate(output_nodes)}
        output_nodes = [output_nodes_map[nid] for nid in seed_nodes]
        sentence_features = []
        tokenized = self.tokenizer(input_text, padding=True, truncation=True, max_length=self.token_length, return_tensors='pt')
        sentence_features.append(tokenized)
        for idx in range(2):
            tokenized = self.tokenizer(texts[idx], padding=True, truncation=True, max_length=self.token_length, return_tensors='pt')
            sentence_features.append(tokenized)
        ret['sentence_features'] = sentence_features
        ret['labels'] = torch.tensor(batch_hard_labels, dtype=torch.long)
        ret['blocks'] = blocks
        return ret


@dataclass
class DataCollatorForSBertNCL:
    token_length: int = 32
    tokenizer: Union[AutoTokenizer, str] = "bert-base-uncased"
    query_path: str = 'data/mt_new/queries'
    spu_path: str = 'data/mt_new/spus'

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        _, self.q2i  = load_token_mapping(self.query_path)
        _, self.s2i = load_token_mapping(self.spu_path)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_hard_labels = []
        batch_soft_labels = []
        texts = [[] for _ in range(2)]
        ret = {}
        query_nodes, spu_nodes = [], []
        for sample in samples:
            query, spu, hard_label, soft_label = sample
            texts[0].append(query)
            texts[1].append(spu)
            batch_hard_labels.append(hard_label)
            if soft_label is not None:
                batch_soft_labels.append(soft_label)
            query_nodes.append(self.q2i[query])
            spu_nodes.append(self.s2i[spu])
        sentence_features = []
        for idx in range(2):
            tokenized = self.tokenizer(texts[idx], padding=True, truncation=True, max_length=self.token_length, return_tensors='pt')
            sentence_features.append(tokenized)
        ret['sentence_features'] = sentence_features
        ret['hard_labels'] = torch.tensor(batch_hard_labels, dtype=torch.long)
        ret['soft_labels'] = torch.tensor(batch_soft_labels, dtype=torch.float) if batch_soft_labels else None
        ret['query_ids'] = torch.tensor(query_nodes, dtype=torch.long)
        ret['spu_ids'] = torch.tensor(spu_nodes, dtype=torch.long)
        return ret

class DataCollatorForXRT:
    def __init__(self, query_path: str, sup_path: str):
        _, self.query2idx = load_token_mapping(query_path)
        idx2sup, _ = load_token_mapping(sup_path)
        self.sup2idx = {sup: i + len(self.query2idx) for i, sup in enumerate(idx2sup)}

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        queries, sups, labels = map(list, zip(*samples))
        batch = {}
        queries = [self.query2idx[q] for q in queries]
        sups = [self.sup2idx[s] for s in sups]
        batch['queries'] = torch.tensor(queries, dtype=torch.long)
        batch['sups'] = torch.tensor(sups, dtype=torch.long)
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        return batch


@dataclass
class DataCollatorForTextGnn:
    token_length: int = 32
    tokenizer: Union[AutoTokenizer, str] = "bert-base-uncased"
    query_neigh_num: int = 3
    spu_neigh_num: int = 3
    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

    def __call__(self, samples: List[List[Union[str, int]]]) -> Dict[str, torch.Tensor]:
        batch_labels = []
        texts = [[] for _ in range(4)]
        ret = {}
        query_neigh_lengths = []
        spu_neigh_lengths = []
        for sample in samples:
            query, spu, label = sample
            query_arr = query.split("'!!@@##$$'")
            spu_arr = spu.split("'!!@@##$$'")

            query_neigh_lengths.append(len(query_arr))
            spu_neigh_lengths.append(len(spu_arr))

            query_arr = query_arr[:self.query_neigh_num+1]

            if len(query_arr)<self.query_neigh_num+1:
                query_neigh_arr = query_arr + [""]*(self.query_neigh_num-len(query_arr))

            if len(spu_arr)<self.spu_neigh_num+1:
                spu_neigh_arr = spu_arr + [""]*(self.spu_neigh_num-len(spu_arr))

            texts[0].append(query)
            texts[1].append(spu)

            texts[2].extend(query_neigh_arr)
            texts[3].extend(spu_neigh_arr)
            
            batch_labels.append(label)

        sentence_features = []
        for idx in range(4):
            # tokenized返回的是一个字典
            tokenized = self.tokenizer(texts[idx], padding=True, truncation=True, max_length=self.token_length, return_tensors='pt')
            sentence_features.append(tokenized)
        ret['sentence_features'] = sentence_features
        ret['query_neigh_lengths'] = torch.tensor(query_neigh_lengths, dtype=torch.long)
        ret['spu_neigh_lengths'] = torch.tensor(spu_neigh_lengths, dtype=torch.long)
        ret['labels'] = torch.tensor(batch_labels, dtype=torch.long)
        return ret


class NeighborSampler(BlockSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace

    def sample_blocks(self, g, seed_nodes, exclude_eids=None, is_neg=False):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            if is_neg:
                neg_src, neg_dst = frontier.edges()
                # 如果sample到自身，那么负样本随机采样
                mask = neg_src == neg_dst
                mask_eids = torch.arange(0, frontier.num_edges())[mask]
                frontier = dgl.remove_edges(frontier, mask_eids)
                random_neg_src = torch.randint(low=0, high=g.num_nodes(), size=neg_src[mask].shape)
                mask_neg_dst = neg_dst[mask]
                frontier = dgl.add_edges(frontier, random_neg_src, mask_neg_dst)
            eid = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier, seed_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

    def sample(self, g, seed_nodes, exclude_eids=None, is_neg=False):     # pylint: disable=arguments-differ
        """Sample a list of blocks from the given seed nodes."""
        result = self.sample_blocks(g, seed_nodes, exclude_eids=exclude_eids, is_neg=is_neg)
        return self.assign_lazy_features(result)

def get_dataset(dataset):
    dic = {
        'bert': BertData,
        'sbert_distill': SBertDataDistill,
        'sbert_distill_q2q': SBertDataDistillQ2q,
        'intent': IntentData,
    }
    return dic[dataset]

def get_data_collator(data_collator):
    dic = {
        'bert': DataCollatorForTuning,
        'sbert': DataCollatorForSBert,
        'xrt': DataCollatorForXRT,
        'textgnn': DataCollatorForTextGnn,
        'sbert_distill': DataCollatorForSBertDistill,
        'sbert_distill_gnn': DataCollatorForSBertGNN,
        'sbert_distill_q2q': DataCollatorForSBertDistillQ2q,
        'sbert_distill_q2s': DataCollatorForSBertDistillQ2s,
        'sbert_distill_cont': DataCollatorForSBertDistillContrastive,
        'sbert_distill_ncl': DataCollatorForSBertNCL,
        'sbert_dgi': DataCollatorForSBertDGI,
        'sbert_png': DataCollatorForSBertPNG,
        'sbert_gnn2': DataCollatorForSBertGNN2,
        'bert_pretrain': DataCollatorForPretraining,
        'intent_cl': DataCollatorForIntentClassification,
        'intent_2t': DataCollatorForIntent2Tower,
        'intent_yj': DataCollatorForICYJ,
        'simcse': DataCollatorForSimcse,
        'gnn': DataCollatorForGNN,
        'icgnn': DC4ICGNN,
        'icbertgnn': DC4ICBERTGNN,
    }
    return dic[data_collator]