import json
from module.data import BertData, DataCollatorForSBert, get_data_collator, get_dataset
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
    train_data_collator = None
    train_data = None
    if config.train_dataset:
        train_data_collator = get_data_collator(config.train_data_collator.name)(**config.train_data_collator.params)
        train_data = get_dataset(config.train_dataset.name)(**config.train_dataset.params)

    val_data_collator = None
    val_data = None
    if config.val_dataset:
        val_data_collator = get_data_collator(config.val_data_collator.name)(**config.val_data_collator.params)
        val_data = get_dataset(config.val_dataset.name)(**config.val_dataset.params)
    
    test_data_collator = None
    test_data = None
    if config.test_dataset:
        test_data_collator = get_data_collator(config.test_data_collator.name)(**config.test_data_collator.params)
        test_data = get_dataset(config.test_dataset.name)(**config.test_dataset.params)
    
    train_loader, val_loader, test_loader = None, None, None
    if train_data is not None:
        if config.distributed_train:
            train_loader = DataLoader(
                    train_data, 
                    batch_size=config.train_batch_size, 
                    collate_fn=train_data_collator,
                    shuffle=False,
                    sampler=DistributedSampler(train_data)
                )
        else:
            train_loader = DataLoader(
                train_data, shuffle=True, batch_size=config.train_batch_size, collate_fn=train_data_collator
            )
    if val_data is not None:
        val_loader = DataLoader(
                val_data, shuffle=False, batch_size=config.valid_batch_size, collate_fn=val_data_collator
            )
    if test_data is not None:
        test_loader = DataLoader(
                test_data, shuffle=False, batch_size=config.test_batch_size, collate_fn=test_data_collator
            )
    solver = Solver

    solver = solver(config, train_loader, val_loader, test_loader)

    solver.build(rank)
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        test_res = solver.eval(0, True)
        solver.logger.info(json.dumps(test_res))
    elif config.mode == 'run_method':
        solver.run_method(**config.method_params)
    if config.distributed_train:
        destroy_process_group()

if __name__ == '__main__':
    config = get_config()
    print(config)
    config.parser.register('type', None, identity)
    if config.distributed_train:
        world_size = torch.cuda.device_count()
        print(world_size)
        config.world_size = world_size
        mp.spawn(main, args=(world_size, config), nprocs=world_size)
    else:
        main(0, 0, config)