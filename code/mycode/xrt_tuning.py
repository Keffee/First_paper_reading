import json
from module.data import BertData, DataCollatorForSBert, DataCollatorForXRT
from solver import Solver
from module import get_config
import os
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import torch

if __name__ == '__main__':
    config = get_config()
    print(config)

    data_collator = DataCollatorForXRT(config.query_path, config.sup_path)
    train_data = BertData(config.train_data_path)
    val_data = BertData(config.valid_data_path)
    test_data = BertData(config.test_data_path)

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

    solver.build()
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        test_res = solver.eval(0, True)
        solver.logger.info(json.dumps(test_res))
