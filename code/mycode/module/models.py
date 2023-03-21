from transformers import BertModel, BertForPreTraining
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, losses
import numpy as np
import torch
import dgl
try:
    from layers import *
    from utils import get_mask_from_lengths
except ImportError:
    from .layers import *
    from .utils import get_mask_from_lengths

################################################
class Intent2Tower(nn.Module):
    def __init__(self, config) -> None:
        super(Intent2Tower, self).__init__()
        self.bert = SentenceTransformer(config.bert_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
        self.bert.max_seq_length = config.model_params.max_length
        self.loss_func = None
        if config.model_params.loss == 'bpr':
            self.loss_func = bpr_loss
        elif config.model_params.loss == 'bce':
            self.loss_func = bce_loss

    def test_loss(self, sentence_features, **kwargs):
        # 输入的句子
        return self.forward(sentence_features) 
    
    def encode(self, sentences):
        return self.bert.encode(sentences, convert_to_tensor=True)

    def forward(self, sentence_features, **kwargs):
        # 输入的句子
        query = self.bert(sentence_features[0])['sentence_embedding']
        pos = self.bert(sentence_features[1])['sentence_embedding']
        neg = self.bert(sentence_features[2])['sentence_embedding']
        pos_score = torch.sum(query * pos, dim=-1)
        neg_score = torch.sum(query * neg, dim=-1)
        return self.loss_func(pos_score, neg_score)

class SimpleCLS(nn.Module):
    def __init__(self, config) -> None:
        super(SimpleCLS, self).__init__()
        emb = torch.tensor(np.load(config.model_params.emb_path), dtype=torch.float)
        emb = torch.cat([emb, torch.rand(1, emb.shape[-1])], dim=0)
        self.emb = nn.Embedding.from_pretrained(emb, freeze=config.model_params.emb_freeze)
        self.classifier = nn.Linear(emb.shape[-1], config.model_params.num_classes)

    def test_loss(self, sentence_features, **kwargs):
        # 输入的句子
        return self.forward(sentence_features) 
    
    def test(self, sentence_features, **kwargs):
        # 输入的句子
        sent_emb = self.emb(sentence_features)
        score = self.classifier(sent_emb)
        return score

    def forward(self, sentence_features, labels, **kwargs):
        # 输入的句子
        # print(max(sentence_features))
        # print(self.emb.weight.shape)
        sent_emb = self.emb(sentence_features)
        score = self.classifier(sent_emb)
        loss = F.cross_entropy(score, labels)
        return loss


class NodeClassify(nn.Module):
    def __init__(self, config):
        super(NodeClassify, self).__init__()
        g, _ = dgl.load_graphs(config.model_params.graph)
        g = g[0]
        self.h_dim = config.model_params.h_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.num_bases = len(self.rel_names)
        self.num_hidden_layers = config.model_params.num_hidden_layers
        self.dropout = config.model_params.dropout
        self.use_self_loop = config.model_params.use_self_loop
        self.classifier = nn.Linear(config.model_params.h_dim, config.model_params.num_classes)
        self.layers = nn.ModuleList()
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(
                RelGraphConvLayer(
                    self.h_dim,
                    self.h_dim,
                    self.rel_names,
                    self.num_bases,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
                )
            )
        self.loss = nn.CrossEntropyLoss()
        freeze = config.model_params.emb_freeze
        self.query_emb = nn.Embedding.from_pretrained(torch.tensor(np.load(config.model_params.query_path)), freeze=freeze)
        self.item_emb = nn.Embedding.from_pretrained(torch.tensor(np.load(config.model_params.item_path)), freeze=freeze)
        self.cate_emb = nn.Embedding.from_pretrained(torch.tensor(np.load(config.model_params.cate_path)), freeze=freeze)

    def test(self, input_nodes, output_nodes, blocks=None, labels=None):
        h = {}
        h['query'] = self.query_emb(input_nodes['query'])
        h['item'] = self.item_emb(input_nodes['item'])
        h['cate'] = self.cate_emb(input_nodes['cate'])
        # minibatch training
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        h = h['query'][output_nodes]
        score = self.classifier(h)
        return score

    def forward(self, input_nodes, output_nodes, blocks=None, labels=None):
        h = {}
        h['query'] = self.query_emb(input_nodes['query'])
        h['item'] = self.item_emb(input_nodes['item'])
        h['cate'] = self.cate_emb(input_nodes['cate'])
        # minibatch training
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        h = h['query'][output_nodes]
        score = self.classifier(h)
        loss = self.loss(score, labels)
        return loss

class NodeBertGNNClassify(nn.Module):
    def __init__(self, config):
        super(NodeBertGNNClassify, self).__init__()
        g, _ = dgl.load_graphs(config.model_params.graph)
        g = g[0]
        self.h_dim = config.model_params.h_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.num_bases = len(self.rel_names)
        self.num_hidden_layers = config.model_params.num_hidden_layers
        self.dropout = config.model_params.dropout
        self.use_self_loop = config.model_params.use_self_loop
        self.classifier = nn.Linear(config.model_params.h_dim, config.model_params.num_classes)
        self.layers = nn.ModuleList()
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(
                RelGraphConvLayer(
                    self.h_dim,
                    self.h_dim,
                    self.rel_names,
                    self.num_bases,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
                )
            )
        self.loss = nn.CrossEntropyLoss()
        self.bert = SentenceTransformer(config.model_params.bert_path)

    def test(self, sentence_features, output_nodes, blocks=None, labels=None):
        h = {}
        h['query'] = self.bert(sentence_features[0])['sentence_embedding']
        h['item'] = self.bert(sentence_features[1])['sentence_embedding']
        h['cate'] = self.bert(sentence_features[2])['sentence_embedding']
        # minibatch training
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        h = h['query'][output_nodes]
        score = self.classifier(h)
        return score

    def forward(self, sentence_features, output_nodes, blocks=None, labels=None):
        h = {}
        h['query'] = self.bert(sentence_features[0])['sentence_embedding']
        h['item'] = self.bert(sentence_features[1])['sentence_embedding']
        h['cate'] = self.bert(sentence_features[2])['sentence_embedding']
        # minibatch training
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        h = h['query'][output_nodes]
        score = self.classifier(h)
        loss = self.loss(score, labels)
        return loss

class NodeClassify2(nn.Module):
    def __init__(self, config):
        super(NodeClassify2, self).__init__()
        g, _ = dgl.load_graphs(config.model_params.graph)
        g = g[0]
        self.h_dim = config.model_params.h_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.num_bases = len(self.rel_names)
        self.num_hidden_layers = config.model_params.num_hidden_layers
        self.dropout = config.model_params.dropout
        self.use_self_loop = config.model_params.use_self_loop
        self.weight = nn.Linear(config.model_params.h_dim, config.model_params.h_dim)
        self.classifier = nn.Linear(config.model_params.h_dim, config.model_params.num_classes)
        self.layers = nn.ModuleList()
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(
                RelGraphConvLayer(
                    self.h_dim,
                    self.h_dim,
                    self.rel_names,
                    self.num_bases,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
                )
            )
        self.loss = nn.CrossEntropyLoss()
        freeze = config.model_params.emb_freeze
        self.query_emb = nn.Embedding.from_pretrained(torch.tensor(np.load(config.model_params.query_path)), freeze=freeze)
        self.item_emb = nn.Embedding.from_pretrained(torch.tensor(np.load(config.model_params.item_path)), freeze=freeze)
        self.cate_emb = nn.Embedding.from_pretrained(torch.tensor(np.load(config.model_params.cate_path)), freeze=freeze)

    def test(self, input_nodes, output_nodes, blocks=None, labels=None):
        h = {}
        h['query'] = self.weight(self.query_emb(input_nodes['query']))
        h['item'] = self.weight(self.item_emb(input_nodes['item']))
        h['cate'] = self.weight(self.cate_emb(input_nodes['cate']))
        # minibatch training
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        h = h['query'][output_nodes]
        score = self.classifier(h)
        return score

    def forward(self, input_nodes, output_nodes, blocks=None, labels=None):
        h = {}
        h['query'] = self.weight(self.query_emb(input_nodes['query']))
        h['item'] = self.weight(self.item_emb(input_nodes['item']))
        h['cate'] = self.weight(self.cate_emb(input_nodes['cate']))
        # minibatch training
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        h = h['query'][output_nodes]
        score = self.classifier(h)
        loss = self.loss(score, labels)
        return loss


################################################
class BertFinetune(nn.Module):
    def __init__(self, config) -> None:
        super(BertFinetune, self).__init__()
        self.bert = BertModel.from_pretrained(
            config.bert_path, output_hidden_states=False
        )
        for param in self.bert.parameters():
            param.requires_grad = True
        self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
    
    def infer(self, input_ids, token_type_ids, attention_mask):
        bert_out = self.bert(input_ids, attention_mask, token_type_ids, return_dict=True)
        pooler_output = bert_out["pooler_output"]
        return pooler_output
    
    def test(self, input_ids, token_type_ids, attention_mask, **kwargs):
        # 输入的句子
        cls_embed = self.infer(input_ids, token_type_ids, attention_mask)
        score = self.classifier(cls_embed)
        return score
    
    def normalized_logits(self, input_ids, token_type_ids, attention_mask, **kwargs):
        score = self.test(input_ids, token_type_ids, attention_mask)
        return F.softmax(score, dim=-1)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        # 输入的句子
        cls_embed = self.infer(input_ids, token_type_ids, attention_mask)
        score = self.classifier(cls_embed)
        loss = F.cross_entropy(score, labels)
        return loss

class BertPretrain(nn.Module):
    def __init__(self, config) -> None:
        super(BertPretrain, self).__init__()
        self.bert = BertForPreTraining.from_pretrained(
            config.bert_path
        )
        for param in self.bert.parameters():
            param.requires_grad = True

    def test_loss(self, input_ids, token_type_ids, attention_mask, labels, next_sentence_label, **kwargs):
        # 输入的句子
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, next_sentence_label=next_sentence_label)
        return output.loss

    def forward(self, input_ids, token_type_ids, attention_mask, labels, next_sentence_label, **kwargs):
        # 输入的句子
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, next_sentence_label=next_sentence_label)
        return output.loss

class SentenceBert(nn.Module):
    def __init__(self, config) -> None:
        super(SentenceBert, self).__init__()
        self.bert = SentenceTransformer(config.bert_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
        self.bert.max_seq_length = config.max_length
        if config.loss_type == "cos":
            print("loss_type cos")
            self.softmax_loss = losses.CosineSimilarityLoss(model=self.bert, loss_fct = nn.BCEWithLogitsLoss())            
        else:
            self.softmax_loss = losses.SoftmaxLoss(model=self.bert, sentence_embedding_dimension=self.bert.get_sentence_embedding_dimension(), num_labels=config.num_classes)

    def test(self, sentence_features, **kwargs):
        # 输入的句子
        _, score = self.softmax_loss(sentence_features, labels=None)
        return score 

    def encode(self, sentences):
        return self.bert.encode(sentences, convert_to_tensor=True, device=self.bert.device, show_progress_bar=True)

    def forward(self, sentence_features, labels):
        # 输入的句子
        return self.softmax_loss(sentence_features, labels)
    
    def get_mis_pred_mask(self, sentence_features, labels, **kwargs):
        _, score = self.softmax_loss(sentence_features, labels=None)
        pred = score.max(-1)[1]
        pred = pred.view(score.shape[0], -1)
        labels = labels.view(score.shape[0], -1)
        return pred == labels

class SentenceBertDistill(nn.Module):
    def __init__(self, config) -> None:
        super(SentenceBertDistill, self).__init__()
        self.bert = SentenceTransformer(config.bert_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
        self.bert.max_seq_length = config.max_length

        self.loss = SBertDistillLoss(model=self.bert, sentence_embedding_dimension=self.bert.get_sentence_embedding_dimension(), num_labels=config.num_classes, hard_label_lambda=config.hard_label_lambda)

    def test(self, sentence_features, **kwargs):
        # 输入的句子
        _, score = self.loss(sentence_features, soft_labels=None, hard_labels=None)
        return score 
    
    def test_loss(self, sentence_features, soft_labels, **kwargs):
        # 输入的句子
        loss = self.loss(sentence_features, soft_labels=soft_labels, hard_labels=None)
        return loss 
    
    def get_mis_pred_mask(self, sentence_features, labels):
        _, score = self.loss(sentence_features, soft_labels=None, hard_labels=None)
        pred = score.max(-1)[1]
        pred = pred.view(score.shape[0], -1)
        labels = labels.view(score.shape[0], -1)
        return pred == labels

    def forward(self, sentence_features, soft_labels, hard_labels):
        # 输入的句子
        return self.loss(sentence_features, soft_labels, hard_labels)


class SentenceBertGNNDistill(nn.Module):
    def __init__(self, config) -> None:
        super(SentenceBertGNNDistill, self).__init__()
        self.bert = SentenceTransformer(config.model_params.bert_path)
        self.embedding = nn.Embedding(config.model_params.num_nodes, config.model_params.gnn_emb)
        if 'gnn' not in config.model_params or config.model_params.gnn == 'gcn':
            self.gnn = GCN(config.model_params.gnn_emb, config.model_params.gnn_layers)
        elif config.model_params.gnn == 'lightgcn':
            self.gnn = LightGCN(config.model_params.gnn_emb, config.model_params.gnn_layers)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
        # self.bert.max_seq_length = config.max_length

        self.loss = SBertDistillGNNLoss(model=self.bert, sentence_embedding_dimension=config.model_params.gnn_emb+self.bert.get_sentence_embedding_dimension(), num_labels=config.num_classes, hard_label_lambda=config.hard_label_lambda)

    def test(self, output_nodes, blocks, sentence_features, **kwargs):
        # 输入的句子
        input_nodes = blocks[0].ndata[dgl.NID]['_N']
        input_feats = self.embedding(input_nodes)
        gnn_rep = self.gnn(blocks, input_feats)
        _, score = self.loss(gnn_rep[output_nodes], sentence_features, soft_labels=None, hard_labels=None)
        return score 
    
    def test_loss(self, output_nodes, blocks, sentence_features, soft_labels, hard_labels=None, **kwargs):
        # 输入的句子
        input_nodes = blocks[0].ndata[dgl.NID]['_N']
        input_feats = self.embedding(input_nodes)
        gnn_rep = self.gnn(blocks, input_feats)
        return self.loss(gnn_rep[output_nodes], sentence_features, soft_labels, hard_labels)

    def forward(self, output_nodes, blocks, sentence_features, soft_labels, hard_labels):
        input_nodes = blocks[0].ndata[dgl.NID]['_N']
        # print(input_nodes.device)
        input_feats = self.embedding(input_nodes)
        gnn_rep = self.gnn(blocks, input_feats)
        return self.loss(gnn_rep[output_nodes], sentence_features, soft_labels, hard_labels)


class GNN(nn.Module):
    def __init__(self, config) -> None:
        super(GNN, self).__init__()
        query_emb = torch.tensor(np.load(config.model_params.query_emb))
        spu_emb = torch.tensor(np.load(config.model_params.spu_emb))
        emb = torch.cat((query_emb, spu_emb), dim=0)
        self.embedding = nn.parameter.Parameter(emb, requires_grad=not config.model_params.freeze_emb)
        if 'gnn' not in config.model_params or config.model_params.gnn == 'gcn':
            self.gnn = GCN(config.model_params.gnn_emb, config.model_params.gnn_layers)
        elif config.model_params.gnn == 'lightgcn':
            self.gnn = LightGCN(config.model_params.gnn_emb, config.model_params.gnn_layers)
        elif config.model_params.gnn == 'gat':
            self.gnn = GAT(config.model_params.gnn_emb, config.model_params.gnn_layers)
        self.classifier = nn.Linear(4 * config.model_params.gnn_emb, config.model_params.num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        if 'dropout' not in config.model_params:
            config.model_params.dropout = 0
        self.dropout = nn.Dropout(config.model_params.dropout)

    def infer(self, output_nodes, blocks, labels):
        input_nodes = blocks[0].ndata[dgl.NID]['_N']
        # print(input_nodes.device)
        input_feats = self.embedding[input_nodes]
        gnn_rep = self.gnn(blocks, input_feats)
        final_nodes = blocks[-1].dstdata[dgl.NID]
        origin_rep = self.embedding[final_nodes]
        gnn_rep = gnn_rep[output_nodes]
        origin_rep = origin_rep[output_nodes]
        nodes_rep = self.dropout(origin_rep + gnn_rep)
        
        query_num = nodes_rep.shape[0]//2
        rep_a = nodes_rep[:query_num]
        rep_b = nodes_rep[query_num:]
        vectors_concat = []
        vectors_concat.append(rep_a)
        vectors_concat.append(rep_b)

        vectors_concat.append(torch.abs(rep_a - rep_b))

        vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)
        

        output = self.classifier(features)
        loss = None
        if labels is not None:
            loss = self.ce_loss(output, labels.view(-1))
        if loss is None:
            return [rep_a, rep_b], output
        else:
            return loss

    def forward(self, output_nodes, blocks, labels):
        return self.infer(output_nodes, blocks, labels)

    def test(self, output_nodes, blocks, **kwargs):
        # 输入的句子
        _, score = self.infer(output_nodes, blocks, labels=None)
        return score 
    
    def test_loss(self, output_nodes, blocks, labels):
        return self.infer(output_nodes, blocks, labels)

class SentenceBertDistillQ2q(nn.Module):
    def __init__(self, config) -> None:
        super(SentenceBertDistillQ2q, self).__init__()
        self.bert = SentenceTransformer(config.model_params.bert_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
        if config.model_params.q2q_loss == 'q2q_margin':
            self.loss = SBertDistillQ2qMarginLoss(
                model=self.bert, 
                sentence_embedding_dimension=self.bert.get_sentence_embedding_dimension(), 
                num_labels=config.model_params.num_classes, 
                hard_label_lambda=config.model_params.hard_label_lambda,
                margin_loss_lambda=config.model_params.margin_loss_lambda,
                margin=config.model_params.margin
            )
        elif config.model_params.q2q_loss == 'q2q_infonce':
            self.loss = SBertDistillQ2qInfoNCELoss(
                model=self.bert, 
                sentence_embedding_dimension=self.bert.get_sentence_embedding_dimension(), 
                num_labels=config.model_params.num_classes, 
                hard_label_lambda=config.model_params.hard_label_lambda,
                infonce_loss_lambda=config.model_params.infonce_loss_lambda,
                temp=config.model_params.temp
            )

    def test(self, sentence_features, **kwargs):
        # 输入的句子
        _, score = self.loss(sentence_features, query_sentence_features=None, soft_labels=None, hard_labels=None)
        return score 
    
    def test_loss(self, sentence_features, soft_labels, **kwargs):
        # 输入的句子
        loss = self.loss(sentence_features, query_sentence_features=None, soft_labels=soft_labels, hard_labels=None)
        return loss 

    def forward(self, sentence_features, query_sentence_features, soft_labels, hard_labels):
        # 输入的句子
        return self.loss(sentence_features, query_sentence_features, soft_labels, hard_labels)


class SentenceBertDistillContrastive(nn.Module):
    def __init__(self, config) -> None:
        super(SentenceBertDistillContrastive, self).__init__()
        self.bert = SentenceTransformer(config.model_params.bert_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
        if config.model_params.loss == 'margin':
            pass
        elif config.model_params.loss == 'infonce':
            self.loss = SBertDistillInfoNCELoss(
                model=self.bert, 
                sentence_embedding_dimension=self.bert.get_sentence_embedding_dimension(), 
                num_labels=config.model_params.num_classes, 
                hard_label_lambda=config.model_params.hard_label_lambda,
                infonce_loss_lambda=config.model_params.infonce_loss_lambda,
                temp=config.model_params.temp
            )

    def test(self, sentence_features, **kwargs):
        # 输入的句子
        _, score = self.loss(sentence_features, query_neighbor_features=None, spu_neighbor_features=None, soft_labels=None, hard_labels=None)
        return score 
    
    def test_loss(self, sentence_features, soft_labels, **kwargs):
        # 输入的句子
        loss = self.loss(sentence_features, query_neighbor_features=None, spu_neighbor_features=None, soft_labels=soft_labels, hard_labels=None)
        return loss 

    def forward(self, sentence_features, query_neighbor_features, spu_neighbor_features, soft_labels, hard_labels):
        # 输入的句子
        return self.loss(sentence_features, query_neighbor_features, spu_neighbor_features, soft_labels, hard_labels)
    
    def get_mis_pred_mask(self, sentence_features, labels, **kwargs):
        _, score = self.loss(sentence_features, query_neighbor_features=None, spu_neighbor_features=None, soft_labels=None, hard_labels=None)
        pred = score.max(-1)[1]
        pred = pred.view(score.shape[0], -1)
        labels = labels.view(score.shape[0], -1)
        return pred == labels


class SBertDistillQ2q2Tower(nn.Module):
    """
        两座塔的参数不共享
    """
    def __init__(self, config) -> None:
        super(SBertDistillQ2q2Tower, self).__init__()
        self.query_tower = SentenceTransformer(config.model_params.bert_path)
        self.sup_tower = SentenceTransformer(config.model_params.bert_path)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
        if config.model_params.q2q_loss == 'q2q_margin':
            # self.loss = SBertDistillQ2qMarginLoss(
            #     model=self.query_tower, 
            #     sentence_embedding_dimension=self.query_tower.get_sentence_embedding_dimension(), 
            #     num_labels=config.model_params.num_classes, 
            #     hard_label_lambda=config.model_params.hard_label_lambda,
            #     margin_loss_lambda=config.model_params.margin_loss_lambda,
            #     margin=config.model_params.margin
            # )
            pass
        elif config.model_params.q2q_loss == 'q2q_infonce':
            self.loss = SBertDistillQ2q2TowerInfoNCELoss(
                model_a=self.query_tower, 
                model_b=self.sup_tower, 
                sentence_embedding_dimension=self.query_tower.get_sentence_embedding_dimension(), 
                num_labels=config.model_params.num_classes, 
                hard_label_lambda=config.model_params.hard_label_lambda,
                infonce_loss_lambda=config.model_params.infonce_loss_lambda,
                temp=config.model_params.temp
            )

    def test(self, sentence_features, **kwargs):
        # 输入的句子
        _, score = self.loss(sentence_features, query_sentence_features=None, soft_labels=None, hard_labels=None)
        return score 
    
    def test_loss(self, sentence_features, soft_labels, **kwargs):
        # 输入的句子
        loss = self.loss(sentence_features, query_sentence_features=None, soft_labels=soft_labels, hard_labels=None)
        return loss 

    def forward(self, sentence_features, query_sentence_features, soft_labels, hard_labels):
        # 输入的句子
        return self.loss(sentence_features, query_sentence_features, soft_labels, hard_labels)


class MLP4XRT(nn.Module):
    def __init__(self, config) -> None:
        super(MLP4XRT, self).__init__()
        embed = np.load(config.model_params.xrt_emb)
        self.embed = nn.parameter.Parameter(torch.tensor(embed, dtype=torch.float), requires_grad=config.model_params.xrt_emb_grad)
        self.mlp = MLP(2 * self.embed.shape[1], 8 * self.embed.shape[1], config.model_params.dropout)
        self.classifier = nn.Linear(2 * self.embed.shape[1], config.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.classifier.weight, gain=gain)
        nn.init.zeros_(self.classifier.bias)
    
    def infer(self, queries, sups):
        queries_emb = self.embed[queries]
        sups_emb = self.embed[sups]
        return self.mlp(torch.cat([queries_emb, sups_emb], dim=-1))
    
    def test(self, queries, sups, **kwargs):
        # 输入的句子
        cls_embed = self.infer(queries, sups)
        score = self.classifier(cls_embed)
        return score 

    def forward(self, queries, sups, labels):
        # 输入的句子
        cls_embed = self.infer(queries, sups)
        score = self.classifier(cls_embed)
        loss = F.cross_entropy(score, labels)
        return loss

class MLP4XRT2(nn.Module):
    def __init__(self, config) -> None:
        super(MLP4XRT2, self).__init__()
        embed = np.load(config.model_params.xrt_emb)
        self.embed = nn.parameter.Parameter(torch.tensor(embed, dtype=torch.float), requires_grad=config.model_params.xrt_emb_grad)
        self.mlp = MLP(2 * self.embed.shape[1], 8 * self.embed.shape[1], config.model_params.dropout)
        self.dropout = nn.Dropout(config.model_params.dropout)
        self.classifier = nn.Linear(2 * self.embed.shape[1], config.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.classifier.weight, gain=gain)
        nn.init.zeros_(self.classifier.bias)
    
    def infer(self, queries, sups):
        queries_emb = self.embed[queries]
        sups_emb = self.embed[sups]
        return self.mlp(self.dropout(torch.cat([queries_emb, sups_emb], dim=-1)))
    
    def test(self, queries, sups, **kwargs):
        # 输入的句子
        cls_embed = self.infer(queries, sups)
        score = self.classifier(self.dropout(cls_embed))
        return score 

    def forward(self, queries, sups, labels):
        # 输入的句子
        cls_embed = self.infer(queries, sups)
        score = self.classifier(self.dropout(cls_embed))
        loss = F.cross_entropy(score, labels)
        return loss

class Textgnn(nn.Module):
    def __init__(self, config) -> None:
        super(Textgnn, self).__init__()
        self.query_neigh_num = config.query_neigh_num
        self.spu_neigh_num = config.spu_neigh_num
        self.bert_hidden_size = config.bert_hidden_size
        self.bert = BertModel.from_pretrained(
            config.bert_path, output_hidden_states=False
        )
        for param in self.bert.parameters():
            param.requires_grad = True

        self.classifier = nn.Linear(2*config.bert_hidden_size, config.num_classes)
    
    def infer(self, sentence_features, query_neigh_lengths, spu_neigh_lengths, **kwargs):
        query_bert_out = self.bert(**sentence_features[0], return_dict=True)
        query_pooler_output = query_bert_out["pooler_output"]

        spu_bert_out = self.bert(**sentence_features[1], return_dict=True)
        spu_pooler_output = spu_bert_out["pooler_output"]

        query_neighbor_bert_out = self.bert(**sentence_features[2], return_dict=True)
        query_neighbor_pooler_output = query_neighbor_bert_out["pooler_output"].contiguous().view(-1, self.query_neigh_num, self.bert_hidden_size)

        spu_neighbor_bert_out = self.bert(**sentence_features[3], return_dict=True)
        spu_neighbor_pooler_output = spu_neighbor_bert_out["pooler_output"].contiguous().view(-1, self.spu_neigh_num, self.bert_hidden_size)

        query_neigh_mask = get_mask_from_lengths(query_neigh_lengths).float()
        spu_neigh_mask = get_mask_from_lengths(spu_neigh_lengths).float()

        #print(query_neighbor_pooler_output.shape, query_pooler_output.unsqueeze(2).shape)
        score_ = torch.matmul(query_neighbor_pooler_output,query_pooler_output.unsqueeze(2))
        t = score_ - torch.unsqueeze(query_neigh_mask, 2) * 1e8
        att = torch.softmax(t, dim=-1)
        query_neigh_fea = torch.sum(att * query_neighbor_pooler_output, 1)
        
        score_ = torch.matmul(spu_neighbor_pooler_output, spu_pooler_output.unsqueeze(2))
        t = score_ - torch.unsqueeze(spu_neigh_mask, 2) * 1e8
        att = torch.softmax(t, dim=-1)
        spu_neigh_fea = torch.sum(att * spu_neighbor_pooler_output, 1)

        fusion_fea = torch.cat([query_neigh_fea, spu_neigh_fea],1)
        score = self.classifier(fusion_fea)
        return score
    
    def test(self, sentence_features, query_neigh_lengths, spu_neigh_lengths, **kwargs):
        query_bert_out = self.bert(**sentence_features[0], return_dict=True)
        query_pooler_output = query_bert_out["pooler_output"]

        spu_bert_out = self.bert(**sentence_features[1], return_dict=True)
        spu_pooler_output = spu_bert_out["pooler_output"]

        query_neighbor_bert_out = self.bert(**sentence_features[2], return_dict=True)
        query_neighbor_pooler_output = query_neighbor_bert_out["pooler_output"].contiguous().view(-1, self.query_neigh_num, self.bert_hidden_size)

        spu_neighbor_bert_out = self.bert(**sentence_features[3], return_dict=True)
        spu_neighbor_pooler_output = spu_neighbor_bert_out["pooler_output"].contiguous().view(-1, self.spu_neigh_num, self.bert_hidden_size)

        query_neigh_mask = get_mask_from_lengths(query_neigh_lengths).float()
        spu_neigh_mask = get_mask_from_lengths(spu_neigh_lengths).float()

        #print(query_neighbor_pooler_output.shape, query_pooler_output.unsqueeze(2).shape)
        score_ = torch.matmul(query_neighbor_pooler_output,query_pooler_output.unsqueeze(2))
        t = score_ - torch.unsqueeze(query_neigh_mask, 2) * 1e8
        att = torch.softmax(t, dim=-1)
        query_neigh_fea = torch.sum(att * query_neighbor_pooler_output, 1)
        
        score_ = torch.matmul(spu_neighbor_pooler_output, spu_pooler_output.unsqueeze(2))
        t = score_ - torch.unsqueeze(spu_neigh_mask, 2) * 1e8
        att = torch.softmax(t, dim=-1)
        spu_neigh_fea = torch.sum(att * spu_neighbor_pooler_output, 1)

        fusion_fea = torch.cat([query_neigh_fea, spu_neigh_fea],1)
        score = self.classifier(fusion_fea)
        return score

    def forward(self, sentence_features, query_neigh_lengths, spu_neigh_lengths, labels, **kwargs):
        query_bert_out = self.bert(**sentence_features[0], return_dict=True)
        query_pooler_output = query_bert_out["pooler_output"]

        spu_bert_out = self.bert(**sentence_features[1], return_dict=True)
        spu_pooler_output = spu_bert_out["pooler_output"]

        query_neighbor_bert_out = self.bert(**sentence_features[2], return_dict=True)
        query_neighbor_pooler_output = query_neighbor_bert_out["pooler_output"].contiguous().view(-1, self.query_neigh_num, self.bert_hidden_size)

        spu_neighbor_bert_out = self.bert(**sentence_features[3], return_dict=True)
        spu_neighbor_pooler_output = spu_neighbor_bert_out["pooler_output"].contiguous().view(-1, self.spu_neigh_num, self.bert_hidden_size)

        query_neigh_mask = get_mask_from_lengths(query_neigh_lengths).float()
        spu_neigh_mask = get_mask_from_lengths(spu_neigh_lengths).float()

        #print(query_neighbor_pooler_output.shape, query_pooler_output.unsqueeze(2).shape)
        # (B x neigh_num x d , B x d x 1 ) --> (B x neigh_num x 1)
        score_ = torch.matmul(query_neighbor_pooler_output,query_pooler_output.unsqueeze(2))
        t = score_ - torch.unsqueeze(query_neigh_mask, 2) * 1e8
        att = torch.softmax(t, dim=-1)
        query_neigh_fea = torch.sum(att * query_neighbor_pooler_output, 1)
        
        score_ = torch.matmul(spu_neighbor_pooler_output, spu_pooler_output.unsqueeze(2))
        t = score_ - torch.unsqueeze(spu_neigh_mask, 2) * 1e8
        att = torch.softmax(t, dim=-1)
        spu_neigh_fea = torch.sum(att * spu_neighbor_pooler_output, 1)

        fusion_fea = torch.cat([query_neigh_fea, spu_neigh_fea],1)
        score = self.classifier(fusion_fea)
        loss = F.cross_entropy(score, labels)
        return loss

class SentenceBertNCLDistill(nn.Module):
    def __init__(self, config) -> None:
        super(SentenceBertNCLDistill, self).__init__()
        self.bert = SentenceTransformer(config.model_params.bert_path)
        self.ncl = NCL(config.model_params.ncl_params)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
        # self.bert.max_seq_length = config.max_length
        self.warm_up_step = config.model_params.ncl_params.warm_up_step
        self.loss = SBertDistillGNNLoss(model=self.bert, sentence_embedding_dimension=config.model_params.ncl_params.embedding_size+self.bert.get_sentence_embedding_dimension(), num_labels=config.model_params.num_classes, hard_label_lambda=config.model_params.hard_label_lambda)

    def test(self, query_ids, spu_ids, sentence_features, **kwargs):
        # 输入的句子
        query_emb, spu_emb, reg_loss, ssl_loss, proto_loss = self.ncl(query_ids, spu_ids)
        _, score = self.loss(torch.cat([query_emb, spu_emb], dim=0), sentence_features, soft_labels=None, hard_labels=None)
        return score 
    
    def test_loss(self, query_ids, spu_ids, sentence_features, soft_labels, hard_labels=None, **kwargs):
        # 输入的句子
        query_emb, spu_emb, reg_loss, ssl_loss, proto_loss = self.ncl(query_ids, spu_ids)
        ce = self.loss(torch.cat([query_emb, spu_emb], dim=0), sentence_features, soft_labels, hard_labels)
        return ce + reg_loss + ssl_loss + proto_loss

    def forward(self, query_ids, spu_ids, sentence_features, soft_labels, hard_labels, **kwargs):
        query_emb, spu_emb, reg_loss, ssl_loss, proto_loss = self.ncl(query_ids, spu_ids)
        ce = self.loss(torch.cat([query_emb, spu_emb], dim=0), sentence_features, soft_labels, hard_labels)
        if kwargs['epoch_i'] < self.warm_up_step:
            return ce + reg_loss + ssl_loss
        else:
            return ce + reg_loss + ssl_loss + proto_loss


class Simcse(nn.Module):
    def __init__(self, config) -> None:
        super(Simcse, self).__init__()
        self.bert = SentenceTransformer(
            config.model_params.bert_path
        )
        self.hard_negative_weight = config.model_params.hard_negative_weight
        # 需要设置为可导
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # 设置温度
        self.sim = Similarity(temp=config.model_params.temp)
        self.loss_fct = nn.CrossEntropyLoss()

    def test(self, sentence_features, **kwargs):
        # 输入的句子
        bert_a, bert_b = self.bert(sentence_features[0])["sentence_embedding"], self.bert(sentence_features[1])["sentence_embedding"]
        score = F.cosine_similarity(bert_a, bert_b)
        return score

    def forward(self, sentence_features):
        num_sent = len(sentence_features)

        # Separate representation
        z1, z2 = self.bert(sentence_features[0])["sentence_embedding"], self.bert(sentence_features[1])["sentence_embedding"]

        # Hard negative
        if num_sent == 3:
            z3 = self.bert(sentence_features[2])["sentence_embedding"]

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        #print("11",cos_sim)
        if num_sent >= 3:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
            #print("22",z1_z3_cos)
        # Calculate loss with hard negatives
        if num_sent == 3:
            # Note that weights are actually logits of weights
            # "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
            z3_weight = self.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(z1.device)
            cos_sim = cos_sim + weights

        labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
        loss = self.loss_fct(cos_sim, labels)
        return loss



class DGI(nn.Module):
    def __init__(self, gnn_emb, gnn_layers):
        super(DGI, self).__init__()
        self.encoder = DGIEncoder(
            gnn_emb, gnn_layers
        )
        self.discriminator = DGIDiscriminator(gnn_emb)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, features):
        positive = self.encoder(g, features, corrupt=False)
        negative = self.encoder(g, features, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2

class SentenceBertDGI(nn.Module):
    def __init__(self, config) -> None:
        super(SentenceBertDGI, self).__init__()
        self.bert = SentenceTransformer(config.model_params.bert_path)
        self.dgi = DGI(config.model_params.gnn_emb, config.model_params.gnn_layers)
        self.dgi_loss_lambda = config.model_params.dgi_loss_lambda
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
        # self.bert.max_seq_length = config.max_length

        self.ce_loss = losses.SoftmaxLoss(model=self.bert, sentence_embedding_dimension=self.bert.get_sentence_embedding_dimension(), num_labels=config.model_params.num_classes)

    def test(self, sentence_features, **kwargs):
        # 输入的句子
        _, score = self.ce_loss(sentence_features, labels=None)
        return score 
    
    def test_loss(self,):
        # 输入的句子
        pass

    def forward(self, blocks, sentence_features, labels):
        node_rep = self.bert(sentence_features[0])['sentence_embedding']
        dgi_loss = self.dgi(blocks, node_rep)
        ce_loss = self.ce_loss(sentence_features[1:], labels)
        # print(input_nodes.device)
        return ce_loss + self.dgi_loss_lambda * dgi_loss


class PNG(nn.Module):
    def __init__(self, gnn_emb, gnn_layers, gnn='gcn'):
        super(PNG, self).__init__()
        self.encoder = PNGEncoder(
            gnn_emb, gnn_layers, gnn
        )
        self.discriminator = PNGDiscriminator(gnn_emb)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, graphs, features, label_feat, output_nodes):
        pos_g, neg_g = graphs
        pos_feat, neg_feat = features
        positive = self.encoder(pos_g, pos_feat)
        negative = self.encoder(neg_g, neg_feat)
        # label_feat = torch.sigmoid(label_feat)
        positive = positive[output_nodes]
        negative = negative[output_nodes]
        positive = self.discriminator(positive, label_feat)
        negative = self.discriminator(negative, label_feat)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2

class SentenceBertPNG(nn.Module):
    def __init__(self, config) -> None:
        super(SentenceBertPNG, self).__init__()
        self.bert = SentenceTransformer(config.model_params.bert_path)
        self.png = PNG(config.model_params.gnn_emb, config.model_params.gnn_layers, config.model_params.gnn)
        self.png_loss_lambda = config.model_params.png_loss_lambda
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
        # self.bert.max_seq_length = config.max_length

        self.ce_loss = losses.SoftmaxLoss(model=self.bert, sentence_embedding_dimension=self.bert.get_sentence_embedding_dimension(), num_labels=config.model_params.num_classes)
    def test(self, sentence_features, **kwargs):
        # 输入的句子
        _, score = self.ce_loss(sentence_features, labels=None)
        return score 
    
    def test_loss(self,):
        # 输入的句子
        pass

    def forward(self, graph_blocks, output_nodes, sentence_features, labels):
        pos_feat = self.bert(sentence_features[0])['sentence_embedding']
        neg_feat = self.bert(sentence_features[1])['sentence_embedding']
        node_rep, node_output = self.ce_loss(sentence_features[2:], None)
        node_rep = torch.cat(node_rep, dim=0)
        ce_loss = F.cross_entropy(node_output, labels.view(-1))
        # print(output_nodes.shape)
        # print(node_rep.shape)
        png_loss = self.png(graph_blocks, [pos_feat, neg_feat], node_rep, output_nodes)
        self.png_loss = self.png_loss_lambda * png_loss.item()
        # print(input_nodes.device)
        return ce_loss + self.png_loss_lambda * png_loss

class SentenceBertGNN2(nn.Module):
    """
        gnn接在bert之后，串行的结构
    """
    def __init__(self, config) -> None:
        super(SentenceBertGNN2, self).__init__()
        self.bert = SentenceTransformer(config.model_params.bert_path)
        if 'gnn' not in config.model_params or config.model_params.gnn == 'gcn':
            self.gnn = GCN(config.model_params.gnn_emb, config.model_params.gnn_layers)
        elif config.model_params.gnn == 'lightgcn':
            self.gnn = LightGCN(config.model_params.gnn_emb, config.model_params.gnn_layers)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(config.bert_hidden_size, config.num_classes)
        # self.bert.max_seq_length = config.max_length
        self.classifier = nn.Linear(4 * self.bert.get_sentence_embedding_dimension(), config.model_params.num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.infonce_lambda = config.model_params.infonce_lambda
        if self.infonce_lambda:
            self.sim = Similarity(temp=config.model_params.temp)


    def infer(self, blocks, output_nodes, sentence_features, labels):
        feat = self.bert(sentence_features[0])['sentence_embedding']
        gnn_repr = self.gnn(blocks, feat)
        gnn_repr_tmp = gnn_repr[output_nodes]
        rep_a, rep_b = self.bert(sentence_features[1])['sentence_embedding'], self.bert(sentence_features[2])['sentence_embedding']
        gnn_repr_a = gnn_repr_tmp[:rep_a.shape[0]]
        gnn_repr_b = gnn_repr_tmp[rep_a.shape[0]:]
        rep_a = rep_a + gnn_repr_a
        rep_b = rep_b + gnn_repr_b
        vectors_concat = []
        vectors_concat.append(rep_a)
        vectors_concat.append(rep_b)

        vectors_concat.append(torch.abs(rep_a - rep_b))

        vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss = None
        if labels is not None:
            loss = self.ce_loss(output, labels.view(-1))
            if self.infonce_lambda:
                cos_sim = self.sim(feat[:blocks[0].num_dst_nodes()].unsqueeze(1), gnn_repr.unsqueeze(0))
                labels = torch.arange(cos_sim.size(0)).long().to(rep_a.device)
                infonce_loss = self.infonce_lambda * self.ce_loss(cos_sim, labels)
                self.infonce_loss = infonce_loss.item()
                loss = loss + infonce_loss
        if loss is None:
            return [rep_a, rep_b], output
        else:
            return loss
    
    def forward(self, blocks, output_nodes, sentence_features, labels):
        return self.infer(blocks, output_nodes, sentence_features, labels)

    def test(self, blocks, output_nodes, sentence_features, **kwargs):
        # 输入的句子
        _, score = self.infer(blocks, output_nodes, sentence_features, labels=None)
        return score 
    
    def test_loss(self,):
        # 输入的句子
        pass

class ReprBert(nn.Module):
    def __init__(self, config) -> None:
        super(ReprBert, self).__init__()
        config = config.model_params
        self.hard_label_lambda = config.hard_label_lambda
        self.query_bert_model = BertModel.from_pretrained(
            config.bert_path
        )
        self.title_bert_model = BertModel.from_pretrained(
            config.bert_path
        )
        self.num_hidden_layers = self.query_bert_model.config.num_hidden_layers
        self.hidden_size = self.query_bert_model.config.hidden_size
        for param in self.query_bert_model.parameters():
            param.requires_grad = True
        for param in self.title_bert_model.parameters():
            param.requires_grad = True
        self.query_pool_weight = nn.parameter.Parameter(torch.zeros(1, config.query_len))
        self.query_mid_pool_weight = nn.parameter.Parameter(torch.zeros(self.num_hidden_layers - 1, 1, config.query_len))
        self.title_pool_weight = nn.parameter.Parameter(torch.zeros(1, config.title_len))
        self.title_mid_pool_weight = nn.parameter.Parameter(torch.zeros(self.num_hidden_layers - 1, 1, config.title_len))
        self.fusion_final_weight = nn.parameter.Parameter(torch.zeros(self.num_hidden_layers * self.hidden_size, self.hidden_size))
        self.mlp = MLP2(config.mlp.hidden_size, config.mlp.hidden_size, config.mlp.dropout, config.mlp.act_f)
        self.classifier = nn.Linear(config.mlp.hidden_size, config.num_classes)
        self.reset_parameters()
    
    def reset_parameters(self,):
        nn.init.trunc_normal_(self.query_pool_weight, std=0.02)
        nn.init.trunc_normal_(self.query_mid_pool_weight, std=0.02)
        nn.init.trunc_normal_(self.title_pool_weight, std=0.02)
        nn.init.trunc_normal_(self.title_mid_pool_weight, std=0.02)
        nn.init.trunc_normal_(self.fusion_final_weight, std=0.02)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def test(self, sentence_features, **kwargs):
        # 输入的句子
        _, score = self.forward(sentence_features, soft_labels=None, hard_labels=None)
        return score 
    
    def test_loss(self, sentence_features, soft_labels, **kwargs):
        # 输入的句子
        return self.forward(sentence_features, soft_labels, hard_labels=None)

    def forward(self, sentence_features, soft_labels, hard_labels):
        fusion_final_output = self.infer(sentence_features)
        output = self.classifier(fusion_final_output)
        loss = None
        if soft_labels is not None:
            # print(soft_labels)
            loss = cross_entropy(output, soft_labels)
        if hard_labels is not None:
            hard_label_loss = self.hard_label_lambda * F.cross_entropy(output, hard_labels.view(-1))
            loss = loss + hard_label_loss if loss else hard_label_loss
        if loss is None:
            return fusion_final_output, output
        else:
            return loss

    def infer(self, sentence_features):
        # query
        mask_query = sentence_features[0]['attention_mask'].unsqueeze(2)
        query_bert_dict = self.query_bert_model(**sentence_features[0], output_hidden_states=True, return_dict=True)

        query_bert_output = query_bert_dict['last_hidden_state'] * mask_query
        query_bert_mid_outputs = query_bert_dict['hidden_states']
        query_bert_mid_outputs = torch.stack(query_bert_mid_outputs[1:-1])

        # spus
        mask_title = sentence_features[1]['attention_mask'].unsqueeze(2)
        title_bert_dict = self.title_bert_model(**sentence_features[1], output_hidden_states=True, return_dict=True)
        title_bert_output = title_bert_dict['last_hidden_state'] * mask_title
        title_bert_mid_outputs = title_bert_dict['hidden_states']
        title_bert_mid_outputs = torch.stack(title_bert_mid_outputs[1:-1])
        
        # 1*1*seq, b*seq*dim -> b*1*dim
        query_pool_output = torch.matmul(self.query_pool_weight.unsqueeze(0), query_bert_output)
        query_pool_output = query_pool_output.squeeze(1)
        title_pool_output = torch.matmul(self.title_pool_weight.unsqueeze(0), title_bert_output)
        title_pool_output = title_pool_output.squeeze(1)
        fusion_bert_output = torch.maximum(title_pool_output, query_pool_output)

        # mid_interaction
        # n-1*1*1*seq, n-1*b*seq*dim -> n-1*b*1*dim
        fusion_mid_query_outputs = torch.matmul(self.query_mid_pool_weight.unsqueeze(1), query_bert_mid_outputs)
        fusion_mid_query_outputs = fusion_mid_query_outputs.squeeze(2)

        fusion_mid_title_outputs = torch.matmul(self.title_mid_pool_weight.unsqueeze(1), title_bert_mid_outputs)
        fusion_mid_title_outputs = fusion_mid_title_outputs.squeeze(2)

        fusion_mid_outputs = torch.maximum(fusion_mid_query_outputs, fusion_mid_title_outputs)
        # n-1 * b * dim -> b * (n-1 * dim)
        fusion_mid_outputs = fusion_mid_outputs.permute(1, 0, 2)
        fusion_mid_outputs = fusion_mid_outputs.reshape(fusion_mid_outputs.shape[0], -1)

        fusion_final_output = torch.matmul(torch.cat([fusion_mid_outputs, fusion_bert_output], dim=-1), self.fusion_final_weight)
        return fusion_final_output

def get_model(model_name):
    model_dict = {
        'bert': BertFinetune,
        'sbert': SentenceBert,
        'mlp4xrt': MLP4XRT,
        'mlp4xrt2': MLP4XRT2,
        'textgnn': Textgnn,
        'sbert_distill': SentenceBertDistill,
        'bert_pretrain': BertPretrain,
        'sbert_distill_q2q': SentenceBertDistillQ2q,
        'sbert_distill_cont': SentenceBertDistillContrastive,
        'sbert_distill_gnn': SentenceBertGNNDistill,
        'sbert_distill_q2q_2t': SBertDistillQ2q2Tower,
        'sbert_distill_ncl': SentenceBertNCLDistill,
        'sbert_dgi': SentenceBertDGI,
        'sbert_png': SentenceBertPNG,
        'sbert_gnn2': SentenceBertGNN2,
        'intent_2t': Intent2Tower,
        'simple_cls': SimpleCLS,
        'simcse': Simcse,
        'gnn': GNN,
        'node_cls': NodeClassify,
        'node_cls2': NodeClassify2,
        'node_bertgnn': NodeBertGNNClassify,
        'reprbert': ReprBert,
    }
    return model_dict[model_name]