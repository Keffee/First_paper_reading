import os
from datetime import datetime
from collections import defaultdict
import torch
from torch import optim
import torch.nn as nn
from pathlib import Path
from chanfig import Config

# debug = True
debug = False
device = torch.device('cuda' if torch.cuda.is_available() and not debug else 'cpu')

project_dir = Path('__file__').parent.parent
save_dir = project_dir / 'ckpt'
save_dir.mkdir(parents=True, exist_ok=True)
class ModelConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_func = 'eval'
        self.find_unused_parameters = False

def get_config():
    config = ModelConfig()
    config = config.parse(default_config="config")
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if config.description:
        save_path = save_dir / config.dataset / f"{config.model}_{config.description}_{time_now}"
    else:
        save_path = save_dir / config.dataset / f"{config.model}_{time_now}"
    save_path.mkdir(parents=True, exist_ok=True)
    config.save_path = str(save_path)
    config.yaml(Path(config.save_path) / 'config.yml')
    return config