# here put the import lib

import functools
import os
import random
import time
from collections import defaultdict
from typing import Union
import numpy as np
import torch
import logging
import transformers
from torch.distributed import init_process_group
from torch import optim
import torch.nn.functional as F
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0, distributed_train=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_min = 0
        self.best_epoch = -1
        self.delta = delta
        self.model_path = f"{save_path}/model.bin"
        self.distributed_train = distributed_train

    def __call__(self, score, epoch, model=None):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.best_epoch = epoch
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}'
                )
            if self.patience != -1 and self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'score increased ({self.score_min:.6f} --> {score:.6f}).  Saving model ...'
            )
        if self.distributed_train:
            torch.save(model.module.state_dict(), self.model_path)
        else:
            torch.save(model.state_dict(), self.model_path)
        self.score_min = score

    def load_model(self):
        return torch.load(self.model_path)


class LogResult:
    def __init__(self):
        self.result = defaultdict(list)
        pass

    def log(self, result: dict):
        for key, value in result.items():
            self.result[key].append(value)

    def log_single(self, key, value):
        self.result[key].append(value)

    def show_str(self):
        print()
        string = ""
        for key, value_lst in self.result.items():
            if key == "gpu_mem":
                continue
            value = np.mean(value_lst)
            try:
                if isinstance(value, int):
                    string += f"{key}:\n{value}\n{max(value_lst)}\n{min(value_lst)}\n"
                else:
                    string += f"{key}:\n{value:.4f}\n{max(value_lst):.4f}\n{min(value_lst):.4f} \n"
            except TypeError:
                print("error")
        print(string)

def load_token_mapping(path):
    idx2token, token2idx = [], {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            idx2token.append(line)
            token2idx[line] = i
    return idx2token, token2idx

def load_item_mapping(f_path):
    idx2token, token2idx = [], {}
    idx2text, text2idx = [], {}
    with open(f_path, 'r') as f:
        for i, line in enumerate(f):
            item, item_t = line.strip().split('\t')
            idx2token.append(item)
            token2idx[item] = i
            idx2text.append(item_t)
            text2idx[item_t] = i
    return idx2token, token2idx, idx2text, text2idx


def timing(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        return_data = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print(
            f'function [{func.__name__}] finished in {int(elapsedTime * 1000)} ms'
        )
        return return_data

    return newfunc


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def get_scheduler(optimizer, scheduler: str, t_total: int, warmup_steps: Union[int, float]=0.05):
    """
    Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
    """
    if scheduler is None:
        return None
    scheduler = scheduler.lower()
    if isinstance(warmup_steps, float):
        warmup_steps = int(warmup_steps * t_total)
    if scheduler == 'constantlr':
        return transformers.get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmuplinear':
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler == 'warmupcosine':
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))

def get_optimizer(optimizer):
    optimizer_dict = {
        'adam': optim.Adam
    }
    return optimizer_dict[optimizer]

def get_activation_function(act_f):
    act_f_dict = {
        'relu': F.relu,
        'leaky_relu': F.leaky_relu,
        'sigmoid': F.sigmoid
    }
    return act_f_dict[act_f] if act_f is not None else None

def get_mask_from_lengths(lengths, max_len=None, use_cuda=True):
    '''
    param:
        lengths --- [Batch_size]
    return:
        mask --- [Batch_size, max_len]
    '''
    batch_size = lengths.shape[0]  
    if max_len is None:
        max_len = torch.max(lengths).item()  

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    if use_cuda:
        ids = ids.cuda()

    # ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).cuda()  ## 实际需要注意device
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)    ## True 或 False

    return mask

## 实例
## batch_size = 4 , 每句话的长度分别是 [2, 4, 3, 2]
# lengths = torch.tensor([2,4,3,2])
# mask = get_mask_from_lengths(lengths)
# print(mask)

# tensor([[False, False,  True,  True],
#         [False, False, False, False],
#         [False, False, False,  True],
#         [False, False,  True,  True]])

def identity(string):
    return string

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12245"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
