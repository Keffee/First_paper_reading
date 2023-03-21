import json
from module.data import BertData, DataCollatorForTuning
from module.utils import ddp_setup, identity
from solver import Solver
from module import get_config
import os
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

def main(rank: int, world_size: int, config):
    if config.distributed_train:
        ddp_setup(rank, world_size)

    data_collator = DataCollatorForTuning(token_length=config.max_length, tokenizer=config.bert_path)
    train_data = BertData(config.train_data_path, config.aug, config.aug_path)
    val_data = BertData(config.valid_data_path)
    test_data = BertData(config.test_data_path)

    if config.distributed_train:
        train_loader = DataLoader(
                train_data, 
                batch_size=config.train_batch_size, 
                collate_fn=data_collator,
                shuffle=False,
                sampler=DistributedSampler(train_data)
            )
    else:
        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=config.train_batch_size, collate_fn=data_collator
        )

    val_loader = DataLoader(
            val_data, shuffle=False, batch_size=config.valid_batch_size, collate_fn=data_collator
        )

    test_loader = DataLoader(
            test_data, shuffle=False, batch_size=config.test_batch_size, collate_fn=data_collator
        )
    solver = Solver

    solver = solver(config, train_loader, val_loader, test_loader)

    solver.build(rank)
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        test_res = solver.eval(0, True)
        solver.logger.info(json.dumps(test_res))
    if config.distributed_train:
        destroy_process_group()

if __name__ == '__main__':
    config = get_config()
    print(config)
    if config.distributed_train:
        world_size = torch.cuda.device_count()
        print(world_size)
        config.world_size = world_size
        config.parser.register('type', None, identity)
        mp.spawn(main, args=(world_size, config), nprocs=world_size)
    else:
        main(0, 0, config)