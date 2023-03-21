import math
from typing import Callable, Dict, Iterable
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, losses
import dgl
import dgl.nn as dglnn

try:
    from utils import get_activation_function
except ImportError:
    from .utils import get_activation_function
#import faiss

class MLP(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.5):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)
        nn.init.xavier_normal_(self.w_2.weight, gain=gain)
        nn.init.zeros_(self.w_1.bias)
        nn.init.zeros_(self.w_2.bias)


class MLP2(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.5, activation_function=None):
        super(MLP2, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act_f = get_activation_function(activation_function)
        self.reset_parameters()

    def forward(self, x):
        h1 = self.w1(x) + self.dropout(x)
        if self.act_f:
            h1 = self.act_f(h1)
        h2 = self.w2(h1) + self.dropout(h1)
        if self.act_f:
            h2 = self.act_f(h2)
        return h2

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        if self.act_f is F.relu:
            gain = nn.init.calculate_gain('relu')
        elif self.act_f is F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        elif self.act_f is None:
            gain = nn.init.calculate_gain('identity')
        else:
            raise NotImplementedError
        nn.init.xavier_uniform_(self.w1.weight, gain=gain)
        nn.init.xavier_uniform_(self.w2.weight, gain=gain)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)


class GCN(nn.Module):
    def __init__(self, features, num_layers):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList([dgl.nn.GraphConv(features, features, activation=F.relu) for _ in range(num_layers)])

    def forward(self, blocks, x):
        for i, layer in enumerate(self.convs):
            x =layer(blocks[i], x)
        return x

class LightGCN(nn.Module):
    def __init__(self, features, num_layers):
        super(LightGCN, self).__init__()
        self.convs = nn.ModuleList([dgl.nn.GraphConv(features, features, weight=False, bias=False) for _ in range(num_layers)])

    def forward(self, blocks, x):
        for i, layer in enumerate(self.convs):
            x =layer(blocks[i], x)
        return x

class GAT(nn.Module):
    def __init__(self, features, num_layers):
        super(GAT, self).__init__()
        num_heads = 4
        self.convs = nn.ModuleList([dgl.nn.GATConv(features, features, num_heads, activation=F.relu) for _ in range(num_layers)])
        self.fc = nn.Linear(features * num_heads, features)

    def forward(self, blocks, x):
        for i, layer in enumerate(self.convs):
            x =layer(blocks[i], x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

def cross_entropy(pred, soft_targets):
    # logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, -1), 1))


class SBertDistillLoss(losses.SoftmaxLoss):
    def __init__(self, model: SentenceTransformer, sentence_embedding_dimension: int, num_labels: int, concatenation_sent_rep: bool = True, concatenation_sent_difference: bool = True, concatenation_sent_multiplication: bool = False, loss_fct: Callable = nn.CrossEntropyLoss(), hard_label_lambda: float = 0.5):
        super().__init__(model, sentence_embedding_dimension, num_labels, concatenation_sent_rep,
                         concatenation_sent_difference, concatenation_sent_multiplication, loss_fct)
        self.hard_label_lambda = hard_label_lambda

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], soft_labels: Tensor, hard_labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding']
                for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss = None
        if soft_labels is not None:
            # print(soft_labels)
            loss = cross_entropy(output, soft_labels)
        if hard_labels is not None:
            loss = loss + \
                self.hard_label_lambda * self.loss_fct(
                    output, hard_labels.view(-1))
        if loss is None:
            return reps, output
        else:
            return loss


class SBertDistillGNNLoss(losses.SoftmaxLoss):
    def __init__(self, model: SentenceTransformer, sentence_embedding_dimension: int, num_labels: int, concatenation_sent_rep: bool = True, concatenation_sent_difference: bool = True, concatenation_sent_multiplication: bool = False, loss_fct: Callable = nn.CrossEntropyLoss(), hard_label_lambda: float = 0.5):
        super().__init__(model, sentence_embedding_dimension, num_labels, concatenation_sent_rep,
                         concatenation_sent_difference, concatenation_sent_multiplication, loss_fct)
        self.hard_label_lambda = hard_label_lambda

    def forward(self, gnn_rep, sentence_features: Iterable[Dict[str, Tensor]], soft_labels: Tensor, hard_labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding']
                for sentence_feature in sentence_features]
        rep_a, rep_b = reps
        gnn_rep_a, gnn_rep_b = gnn_rep[:rep_a.shape[0]], gnn_rep[rep_a.shape[0]:]
        rep_a = torch.cat([rep_a, gnn_rep_a], dim=-1)
        rep_b = torch.cat([rep_b, gnn_rep_b], dim=-1)
        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss = None
        if soft_labels is not None:
            # print(soft_labels)
            loss = cross_entropy(output, soft_labels)
        if hard_labels is not None:
            hard_label_loss = self.hard_label_lambda * self.loss_fct(output, hard_labels.view(-1))
            loss = loss + hard_label_loss if loss else hard_label_loss
        if loss is None:
            return reps, output
        else:
            return loss

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class SBertDistillQ2qMarginLoss(losses.SoftmaxLoss):
    def __init__(self, model: SentenceTransformer, sentence_embedding_dimension: int, num_labels: int, concatenation_sent_rep: bool = True, concatenation_sent_difference: bool = True, concatenation_sent_multiplication: bool = False, loss_fct: Callable = nn.CrossEntropyLoss(), hard_label_lambda: float = 0.5, margin_loss_lambda = 1, margin = 0.7):
        super().__init__(model, sentence_embedding_dimension, num_labels, concatenation_sent_rep,
                         concatenation_sent_difference, concatenation_sent_multiplication, loss_fct)
        self.hard_label_lambda = hard_label_lambda
        self.margin_loss_lambda = margin_loss_lambda
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], query_sentence_features, soft_labels: Tensor, hard_labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding']
                for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss = None
        if soft_labels is not None:
            # print(soft_labels)
            loss = cross_entropy(output, soft_labels)
        if hard_labels is not None:
            hard_label_loss = self.hard_label_lambda * self.loss_fct(output, hard_labels.view(-1))
            loss = loss + hard_label_loss if loss else hard_label_loss
                
        if query_sentence_features is not None:
            pos_rep, neg_rep = [self.model(sentence_feature)['sentence_embedding']
                for sentence_feature in sentence_features]
            pos_cos = F.cosine_similarity(rep_a, pos_rep, dim=-1)
            neg_cos = F.cosine_similarity(rep_a, neg_rep, dim=-1)
            margin_loss = self.margin_loss_lambda * self.margin_loss(pos_cos, neg_cos, torch.ones(rep_a.shape[0], device=pos_cos.device))
            loss = loss + margin_loss if loss else margin_loss
        if loss is None:
            return reps, output
        else:
            return loss


class SBertDistillQ2qInfoNCELoss(losses.SoftmaxLoss):
    def __init__(self, model: SentenceTransformer, sentence_embedding_dimension: int, num_labels: int, concatenation_sent_rep: bool = True, concatenation_sent_difference: bool = True, concatenation_sent_multiplication: bool = False, loss_fct: Callable = nn.CrossEntropyLoss(), hard_label_lambda: float = 0.5, infonce_loss_lambda = 1, temp = 0.05):
        super().__init__(model, sentence_embedding_dimension, num_labels, concatenation_sent_rep,
                         concatenation_sent_difference, concatenation_sent_multiplication, loss_fct)
        self.hard_label_lambda = hard_label_lambda
        self.infonce_loss_lambda = infonce_loss_lambda
        self.sim = Similarity(temp=temp)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], query_sentence_features, soft_labels: Tensor, hard_labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding']
                for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss = None
        if soft_labels is not None:
            # print(soft_labels)
            loss = cross_entropy(output, soft_labels)
        if hard_labels is not None:
            hard_label_loss = self.hard_label_lambda * self.loss_fct(output, hard_labels.view(-1))
            loss = loss + hard_label_loss if loss else hard_label_loss
                
        if query_sentence_features is not None:
            pos_rep = self.model(query_sentence_features[0])['sentence_embedding']

            # pos_cos = F.cosine_similarity(rep_a, pos_rep, dim=-1)
            cos_sim = self.sim(rep_a.unsqueeze(1), pos_rep.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(rep_a.device)
            # loss_fct = nn.CrossEntropyLoss()
            # print(cos_sim.shape, labels.shape)
            infonce_loss = self.infonce_loss_lambda * F.cross_entropy(cos_sim, labels)
            self.infonce_loss = infonce_loss.item()
            loss = loss + infonce_loss if loss else infonce_loss
        if loss is None:
            return reps, output
        else:
            return loss


class SBertDistillInfoNCELoss(losses.SoftmaxLoss):
    def __init__(self, model: SentenceTransformer, sentence_embedding_dimension: int, num_labels: int, concatenation_sent_rep: bool = True, concatenation_sent_difference: bool = True, concatenation_sent_multiplication: bool = False, loss_fct: Callable = nn.CrossEntropyLoss(), hard_label_lambda: float = 0.5, infonce_loss_lambda = 1, temp = 0.05):
        super().__init__(model, sentence_embedding_dimension, num_labels, concatenation_sent_rep,
                         concatenation_sent_difference, concatenation_sent_multiplication, loss_fct)
        self.hard_label_lambda = hard_label_lambda
        self.infonce_loss_lambda = infonce_loss_lambda
        self.sim = Similarity(temp=temp)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], query_neighbor_features, spu_neighbor_features, soft_labels: Tensor, hard_labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding']
                for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss = None
        if soft_labels is not None:
            # print(soft_labels)
            loss = cross_entropy(output, soft_labels)
        if hard_labels is not None:
            hard_label_loss = self.hard_label_lambda * self.loss_fct(output, hard_labels.view(-1))
            loss = loss + hard_label_loss if loss else hard_label_loss
        self.infonce_loss = 0
        if query_neighbor_features is not None:
            pos_rep = self.model(query_neighbor_features[0])['sentence_embedding']

            # pos_cos = F.cosine_similarity(rep_a, pos_rep, dim=-1)
            cos_sim = self.sim(rep_a.unsqueeze(1), pos_rep.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(rep_a.device)
            # loss_fct = nn.CrossEntropyLoss()
            # print(cos_sim.shape, labels.shape)
            query_infonce_loss = self.infonce_loss_lambda * F.cross_entropy(cos_sim, labels)
            self.infonce_loss += query_infonce_loss.item()
            loss = loss + query_infonce_loss if loss else query_infonce_loss
        if spu_neighbor_features is not None:
            pos_rep = self.model(spu_neighbor_features[0])['sentence_embedding']

            # pos_cos = F.cosine_similarity(rep_a, pos_rep, dim=-1)
            cos_sim = self.sim(rep_b.unsqueeze(1), pos_rep.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(rep_b.device)
            # loss_fct = nn.CrossEntropyLoss()
            # print(cos_sim.shape, labels.shape)
            spu_infonce_loss = self.infonce_loss_lambda * F.cross_entropy(cos_sim, labels)
            self.infonce_loss += spu_infonce_loss.item()
            loss = loss + spu_infonce_loss if loss else spu_infonce_loss
        if loss is None:
            return reps, output
        else:
            return loss


class SBertDistillQ2q2TowerInfoNCELoss(nn.Module):
    def __init__(self, model_a: SentenceTransformer, model_b, sentence_embedding_dimension: int, num_labels: int, concatenation_sent_rep: bool = True, concatenation_sent_difference: bool = True, concatenation_sent_multiplication: bool = False, loss_fct: Callable = nn.CrossEntropyLoss(), hard_label_lambda: float = 0.5, infonce_loss_lambda = 1, temp = 0.05):
        super(SBertDistillQ2q2TowerInfoNCELoss, self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)
        self.loss_fct = loss_fct
        self.hard_label_lambda = hard_label_lambda
        self.infonce_loss_lambda = infonce_loss_lambda
        self.sim = Similarity(temp=temp)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], query_sentence_features, soft_labels: Tensor, hard_labels: Tensor):
        rep_a, rep_b = self.model_a(sentence_features[0])['sentence_embedding'], self.model_b(sentence_features[1])['sentence_embedding']
        reps = [rep_a, rep_b]
        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss = None
        if soft_labels is not None:
            # print(soft_labels)
            loss = cross_entropy(output, soft_labels)
        if hard_labels is not None:
            hard_label_loss = self.hard_label_lambda * self.loss_fct(output, hard_labels.view(-1))
            loss = loss + hard_label_loss if loss else hard_label_loss
                
        if query_sentence_features is not None:
            pos_rep = self.model_a(query_sentence_features[0])['sentence_embedding']

            # pos_cos = F.cosine_similarity(rep_a, pos_rep, dim=-1)
            cos_sim = self.sim(rep_a.unsqueeze(1), pos_rep.unsqueeze(0))
            labels = torch.arange(cos_sim.size(0)).long().to(rep_a.device)
            # loss_fct = nn.CrossEntropyLoss()
            # print(cos_sim.shape, labels.shape)
            infonce_loss = self.infonce_loss_lambda * F.cross_entropy(cos_sim, labels)
            self.infonce_loss = infonce_loss.item()
            loss = loss + infonce_loss if loss else infonce_loss
        if loss is None:
            return reps, output
        else:
            return loss


def bpr_loss(pos, neg):
    loss = F.logsigmoid(pos - neg)
    return -loss.mean()

def bce_loss(pos, neg):
    criterion = F.binary_cross_entropy_with_logits
    # print(pos.shape, neg.shape)
    loss = criterion(pos, torch.ones_like(pos)) + criterion(neg, torch.zeros_like(neg))
    return loss

class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class NCL(nn.Module):

    def __init__(self, config):
        super(NCL, self).__init__()

        # load parameters info
        self.latent_dim = config.embedding_size  # int type: the embedding size of the base model
        self.n_layers = config.n_layers          # int type: the layer num of the base model
        self.reg_weight = config.reg_weight      # float32 type: the weight decay for l2 normalization

        self.ssl_temp = config.ssl_temp
        self.ssl_reg = config.ssl_reg
        self.hyper_layers = config.hyper_layers

        self.alpha = config.alpha

        self.proto_reg = config.proto_reg
        self.k = config.num_clusters
        graph, _ = dgl.load_graphs(config.graph_path)
        self.graph = graph[0]
        self.lightgcn = dgl.nn.GraphConv(self.latent_dim, self.latent_dim, weight=False, bias=False)
        self.n_users = config.n_users
        self.n_items = config.n_items
        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None


        # parameters initialization
        self.reset_parameters()
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None
    
    def reset_parameters(self,):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def e_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()
        device = self.user_embedding.weight.device
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings, device)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings, device)

    def run_kmeans(self, x, device):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(device)
        return centroids, node2cluster

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        pass
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def infer(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers*2)):
            all_embeddings = self.lightgcn(self.graph, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers+1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def ProtoNCE_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_embeddings = user_embeddings_all[user]     # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[user]     # [B,]
        user2centroids = self.user_centroids[user2cluster]   # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding, [self.n_users, self.n_items])

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def forward(self, user_ids, item_ids):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        self.graph = self.graph.to(user_ids.device)
        user = user_ids
        pos_item = item_ids

        user_all_embeddings, item_all_embeddings, embeddings_list = self.infer()

        center_embedding = embeddings_list[0]
        context_embedding = embeddings_list[self.hyper_layers * 2]

        ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, user, pos_item)
        proto_loss = self.ProtoNCE_loss(center_embedding, user, pos_item)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings)

        return user_all_embeddings[user], item_all_embeddings[pos_item], self.reg_weight * reg_loss, ssl_loss, proto_loss

    def predict(self, user_ids, item_ids):
        user, item = user_ids, item_ids

        user_all_embeddings, item_all_embeddings, embeddings_list = self.infer()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, user_ids, **kwargs):
        user = user_ids
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list = self.infer()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)


class DGIEncoder(nn.Module):
    def __init__(self, gnn_emb, gnn_layers):
        super(DGIEncoder, self).__init__()
        self.conv = GCN(gnn_emb, gnn_layers)

    def forward(self, g, features, corrupt=False):
        print(features.shape)
        if corrupt:
            num_nodes = features.shape[0]
            perm = torch.randperm(num_nodes)
            features = features[perm]
        features = self.conv(g, features)
        return features


class DGIDiscriminator(nn.Module):
    def __init__(self, n_hidden):
        super(DGIDiscriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class PNGEncoder(nn.Module):
    def __init__(self, gnn_emb, gnn_layers, gnn='gcn'):
        super(PNGEncoder, self).__init__()
        if gnn == 'lightgcn':
            self.conv = LightGCN(gnn_emb, gnn_layers)
        elif gnn == 'gcn':
            self.conv = GCN(gnn_emb, gnn_layers)


    def forward(self, g, features):
        features = self.conv(g, features)
        return features


class PNGDiscriminator(nn.Module):
    def __init__(self, n_hidden):
        super(PNGDiscriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features_a, features_b):
        features = torch.sum(features_a * torch.matmul(features_b, self.weight), dim=-1)
        return features

class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        rel_names,
        num_bases,
        *,
        weight=True,
        bias=True,
        activation=None,
        self_loop=False,
        dropout=0.0
    ):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    in_feat, out_feat, norm="right", weight=False, bias=False
                )
                for rel in rel_names
            }
        )

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names)
                )
            else:
                self.weight = nn.Parameter(
                    torch.Tensor(len(self.rel_names), in_feat, out_feat)
                )
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain("relu")
                )

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {
                self.rel_names[i]: {"weight": w.squeeze(0)}
                for i, w in enumerate(torch.split(weight, 1, dim=0))
            }
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {
                k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
            }
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}