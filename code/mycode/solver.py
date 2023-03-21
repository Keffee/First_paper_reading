import json
import numpy as np
import torch
import os
from tqdm import tqdm
from math import isnan
import re
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from module import device, seed_torch
from module.models import get_model
from module.utils import EarlyStopping, get_logger, get_optimizer, get_scheduler
from transformers import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from scipy.stats import spearmanr


class Solver(object):
    def __init__(
        self, config, train_data_loader, val_data_loader, test_data_loader
    ):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.writer = None
        self.model = None
        self.scheduler = None
        self.early_stop = None

    #@time_desc_decorator('Build Graph')
    def build(self, rank):
        seed_torch(self.config.random_seed)
        # gpus = ','.join([str(i) for i in self.config.gpus])
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        if self.model is None:
            self.model = get_model(self.config.model)(self.config)
        if self.config.checkpoint:
            self.load_model(self.config.checkpoint)
        self.logger = get_logger(f'{self.config.save_path}/log.log')
        self.distributed_train = self.config.distributed_train
        if self.distributed_train:
            self.device = rank
            self.model = DDP(self.model.to(self.device), device_ids=[self.device], find_unused_parameters=self.config.find_unused_parameters)
        else:
            self.device = device
            self.model.to(self.device)

        # Overview Parameters
        if not self.distributed_train or rank == 0:
            self.logger.info('Model Parameters')
            for name, param in self.model.named_parameters():
                self.logger.info(
                    f"{name}\t{list(param.size())}\trequires_grad={param.requires_grad}"
                )

        if self.config.mode == 'train':
            # self.writer = TensorboardWriter(self.config.logdir)
            if self.config.optimizer_func:
                getattr(self, self.config.optimizer_func)()
            else:
                if self.config.model in ['bert', 'sbert', 'textgnn', 'sbert_distill', 'bert_pretrain', 'sbert_distill_gnn', 'sbert_distill_q2q', 'sbert_distill_q2q_2t', 'intent_2t']:
                    self.optimizer_bert()
                else:
                    self.optimizer = get_optimizer(self.config.optimizer)(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        **self.config.optimizer_params
                    )
            self.scheduler = None
            if 'scheduler_params' in self.config:
                self.config.scheduler_params.t_total = len(self.train_data_loader) * self.config.n_epoch
                self.scheduler = get_scheduler(self.optimizer, **self.config.scheduler_params)
    
    def optimizer_bert(self,):
        param_optimizer = list(
            self.model.named_parameters()
        )  # 模型参数名字列表
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n,
                    p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.01
            },
            {
                'params': [
                    p for n,
                    p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0
            }
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            **self.config.optimizer_params
        )  # To reproduce BertAdam specific behavior set correct_bias=False
    
    def optimizer_sbertgnn(self,):
        if self.distributed_train:
            model = self.model.module
        else:
            model = self.model
        param_optimizer = list(
            model.bert.named_parameters()
        )  # 模型参数名字列表
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n,
                    p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.01
            },
            {
                'params': [
                    p for n,
                    p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0
            },
            {
                'params': model.embedding.parameters(), 
                'lr': 1e-4,
            },
            {
                'params': model.gnn.parameters(), 
                'lr': 1e-4,
            },
            {
                'params': model.loss.classifier.parameters(), 
                'lr': 1e-4,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            **self.config.optimizer_params
        )  # To reproduce BertAdam specific behavior set correct_bias=False
    
    def optimizer_nodebertgnn(self,):
        if self.distributed_train:
            model = self.model.module
        else:
            model = self.model
        param_optimizer = list(
            model.bert.named_parameters()
        )  # 模型参数名字列表
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n,
                    p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.01
            },
            {
                'params': [
                    p for n,
                    p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0
            },
            {
                'params': model.classifier.parameters(), 
                'lr': 1e-3,
            },
            {
                'params': model.layers.parameters(), 
                'lr': 1e-3,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            **self.config.optimizer_params
        )  # To reproduce BertAdam specific behavior set correct_bias=False

    def save_model(self, epoch):
        """Save parameters to checkpoint"""
        ckpt_path = os.path.join(self.config.save_path, f'{epoch}.bin')
        print(f'Save parameters to {ckpt_path}')
        if self.distributed_train:
            torch.save(self.model.module.state_dict(), ckpt_path)
        else:
            torch.save(self.model.state_dict(), ckpt_path)

    def load_model(self, checkpoint, from_parallel=False):
        """Load parameters from checkpoint"""
        print(f'Load parameters from {checkpoint}')
        epoch = re.match(r"[0-9]*", os.path.basename(checkpoint)).group(0)
        if epoch:
            self.epoch_i = int(epoch)
        model_state_dict = torch.load(checkpoint)
        self.model.load_state_dict(model_state_dict, strict=False)

    def write_val_summary(self, epoch_i, res_dict):
        for k, v in res_dict.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    self.writer.add_scalar(f"{k}/{k2}", v2, epoch_i)
            else:
                self.writer.add_scalar(k, v, epoch_i)

    def batch_to_device(self, batch):
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
                elif isinstance(v, list):
                    batch[k] = list(map(lambda x: self.batch_to_device(x), v))
                elif isinstance(v, dict):
                    batch[k] = self.batch_to_device(v)
        elif isinstance(batch, list):
            batch = list(map(lambda x: self.batch_to_device(x), batch))
        else:
            # print(batch)
            batch = batch.to(self.device)
        return batch

    #@time_desc_decorator('Training Start!')
    def train(self, ):
        eval_func = getattr(self, self.config.eval_func)
        is_master = not self.distributed_train or self.device == 0
        if is_master:
            self.writer = SummaryWriter(self.config.save_path)
            self.early_stop = EarlyStopping(self.config.save_path, self.config.early_stop, distributed_train=self.distributed_train)
        epoch_loss_history = []
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = []
            self.model.train()
            if self.distributed_train:
                self.train_data_loader.sampler.set_epoch(epoch_i)
            if self.config.model == 'sbert_distill_ncl' and epoch_i % self.config.model_params.ncl_params.m_step == 0:
                if self.distributed_train:
                    self.model.module.ncl.e_step()
                else:
                    self.model.ncl.e_step()
            for batch_i, batch in enumerate(tqdm(self.train_data_loader)):
                batch = self.batch_to_device(batch)
                if self.config.model == 'sbert_distill_ncl':
                    batch['epoch_i'] = epoch_i
                loss = self.model(**batch)  # 得到预测结果
                assert not isnan(loss.to(torch.device('cpu')).item())
                batch_loss_history.append(loss.item())
                if is_master and batch_i % self.config.print_every == 0:
                    tqdm.write(
                        f'Epoch: {epoch_i}, iter {batch_i}: loss = {loss.item():.3f}'
                    )
                    self.writer.add_scalar(
                        'train_loss',
                        loss.item(),
                        epoch_i * len(self.train_data_loader) + batch_i
                    )
                    # debug
                    tmp_model = self.model.module if self.distributed_train else self.model
                    if 'sbert_distill_q2q' in self.config.model and hasattr(tmp_model.loss, 'infonce_loss'):
                        self.writer.add_scalar(
                            'infonce_loss',
                            tmp_model.loss.infonce_loss,
                            epoch_i * len(self.train_data_loader) + batch_i
                        )
                    if 'sbert_png' in self.config.model and hasattr(tmp_model, 'png_loss'):
                        self.writer.add_scalar(
                            'png_loss',
                            tmp_model.png_loss,
                            epoch_i * len(self.train_data_loader) + batch_i
                        )
                    if 'sbert_gnn2' in self.config.model and hasattr(tmp_model, 'infonce_loss'):
                        self.writer.add_scalar(
                            'infonce_loss',
                            tmp_model.infonce_loss,
                            epoch_i * len(self.train_data_loader) + batch_i
                        )

                # reset gradient
                self.optimizer.zero_grad()
                # Back-propagation
                loss.backward()

                # Gradient cliping
                if 'max_grad_norm' in self.config:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                # Run optimizer
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            epoch_loss = np.sum(batch_loss_history
                                ) / len(batch_loss_history)
            epoch_loss_history.append(epoch_loss)
            self.epoch_loss = epoch_loss
            train_stop_flag = torch.zeros(1).to(self.device)
            if is_master:
                print_str = f'Epoch {epoch_i} loss average: {epoch_loss:.3f}'
                self.logger.info(print_str)
                if epoch_i % self.config.save_every_epoch == 0:
                    self.save_model(epoch_i)
                if self.val_data_loader and epoch_i % self.config.eval_every_epoch == 0:
                    self.logger.info('<Validation>...')
                    val_report = eval_func(epoch_i)
                    self.logger.info(json.dumps(val_report))
                    self.early_stop(val_report[self.config.main_metric], epoch_i, self.model)
                    if self.early_stop.counter == 0 and self.test_data_loader:
                        self.logger.info('<Test>...')
                        test_report = eval_func(epoch_i, True)
                        self.logger.info(json.dumps(test_report))
                if self.early_stop.early_stop:
                    train_stop_flag += 1
                    # break
            if self.distributed_train:
                dist.all_reduce(train_stop_flag, op=dist.ReduceOp.SUM)
            if train_stop_flag == 1:
                break
        if is_master:
            self.logger.info(f"best_val_epoch: {self.early_stop.best_epoch}, best_score: {self.early_stop.best_score}")
        return epoch_loss_history

    def eval(self, epoch_i, test=False):
        self.model.eval()
        self.eval_loss = 0.0
        data_loader = self.test_data_loader if test else self.val_data_loader
        preds = []
        labels = []
        report = None
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch = self.batch_to_device(batch)
            with torch.no_grad():
                if self.distributed_train:
                    scores = self.model.module.test(**batch)
                else:
                    scores = self.model.test(**batch)
            if 'labels' in batch:
                batch_labels = batch['labels']
            else:
                batch_labels = batch['hard_labels']
            labels += batch_labels.flatten().tolist()
            pred = scores.max(-1)[1]
            preds += pred.flatten().tolist()
        report = {
            'auc': metrics.roc_auc_score(labels, preds),
            'accuracy': metrics.accuracy_score(labels, preds),
            'precision': metrics.precision_score(labels, preds, average='binary'),
            'recall': metrics.recall_score(labels, preds, average='binary'),
            'f1': metrics.f1_score(labels, preds, average='binary'),
        }
        if not test:
            self.write_val_summary(epoch_i, report)
        if test:
            pass
        else:
            pass
        return report
    

    def eval_multi_class(self, epoch_i, test=False):
        self.model.eval()
        self.eval_loss = 0.0
        data_loader = self.test_data_loader if test else self.val_data_loader
        preds = []
        labels = []
        report = None
        if test:
            saved_scores = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch = self.batch_to_device(batch)
            with torch.no_grad():
                if self.distributed_train:
                    scores = self.model.module.test(**batch)
                else:
                    scores = self.model.test(**batch)
            batch_labels = batch['labels']
            labels += batch_labels.flatten().tolist()
            pred = scores.max(-1)[1]
            preds += pred.flatten().tolist()
            if test:
                saved_scores.append(scores)
        report = metrics.classification_report(labels, preds, output_dict=True)
        report['f1'] = report['macro avg']['f1-score']
        # report = {
        #     'accuracy': metrics.accuracy_score(labels, preds),
        #     'precision': metrics.precision_score(labels, preds, average='binary'),
        #     'recall': metrics.recall_score(labels, preds, average='binary'),
        #     'f1': metrics.f1_score(labels, preds, average='binary'),
        # }
        if test:
            saved_scores = torch.cat(saved_scores, dim=0)
            np.save(f'{self.config.save_path}/test_scores.npy', saved_scores.cpu().numpy())
        else:
            pass
        return report
    
    def eval_loss(self, epoch_i, test=False):
        self.model.eval()
        data_loader = self.test_data_loader if test else self.val_data_loader
        report = None
        eval_loss = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch = self.batch_to_device(batch)
            with torch.no_grad():
                if self.distributed_train:
                    loss = self.model.module.test_loss(**batch)
                else:
                    loss = self.model.test_loss(**batch)
            eval_loss.append(loss.to('cpu').item())
        report = {
            'loss': -sum(eval_loss) / len(eval_loss)
        }
        if not test:
            self.write_val_summary(epoch_i, report)
        if test:
            pass
        else:
            pass
        return report

    def eval_spearmanr(self, epoch_i, test=False):
        self.model.eval()
        data_loader = self.test_data_loader if test else self.val_data_loader
        report = None
        eval_loss = []
        labels = []
        scores_list = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch = self.batch_to_device(batch)
            with torch.no_grad():
                if self.distributed_train:
                    scores = self.model.module.test(**batch)
                else:
                    scores = self.model.test(**batch)
                
                scores = scores.flatten().tolist()
                #print(batch.keys())
                if 'labels' in batch:
                    labels.extend(batch['labels'].flatten().tolist())
                else:
                    labels.extend([1]*(len(scores)//2)+[0]*(len(scores)//2))
                scores_list.extend(scores)
        
        preds = [ 1 if s>0 else 0 for s in scores_list]
        report = {
            'spearmanr': spearmanr(np.array(labels), np.array(scores_list)).correlation,
            'accuracy': metrics.accuracy_score(labels, preds),
            'precision': metrics.precision_score(labels, preds, average='binary'),
            'recall': metrics.recall_score(labels, preds, average='binary'),
            'f1': metrics.f1_score(labels, preds, average='binary'),
        }


        if not test:
            self.write_val_summary(epoch_i, report)
        if test:
            pass
        else:
            pass
        return report

    def test_case_study(self):
        pass

    def run_method(self, method_name, dataloader, save_name):
        """
            跑一些模型方法然后保存结果
        """
        self.model.eval()
        if dataloader == 'train':
            data_loader = self.train_data_loader
        elif dataloader == 'val':
            data_loader = self.val_data_loader
        elif dataloader == 'test':
            data_loader = self.test_data_loader
        results = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch = self.batch_to_device(batch)
            with torch.no_grad():
                if self.distributed_train:
                    ret = getattr(self.model.module, method_name)(**batch)
                else:
                    ret = getattr(self.model, method_name)(**batch)
            results.append(ret)
        results = torch.cat(results, dim=0)
        results = results.cpu().numpy()
        with open(f'{self.config.save_path}/{save_name}.npy', 'wb') as f:
            np.save(f, results)
