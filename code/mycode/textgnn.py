from module.data import BertData, DataCollatorForTuning, DataCollatorForTextGnn
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

    config.yaml(Path(config.save_path) / 'config.yml')
    data_collator = DataCollatorForTextGnn(token_length=config.max_length, tokenizer=config.bert_path, query_neigh_num=config.query_neigh_num, spu_neigh_num=config.spu_neigh_num)
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
    solver.train()
