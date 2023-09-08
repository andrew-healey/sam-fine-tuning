import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from .cfg import Config
from .models import WrappedSamModel

import numpy as np

def get_optimizer(cfg: Config, sam: WrappedSamModel):
    train = cfg.train
    combined_params = sam.get_trainable_parameters()
    optimizer = optim.AdamW(combined_params, lr=train.initial_lr, betas=(0.9, 0.999), weight_decay=train.weight_decay,_allow_empty_param_list=True)

    num_params = sum([np.prod(p.size()) for p in combined_params if p.requires_grad])
    print(f"Total trainable parameters: {num_params}")

    # Learning rate warmup schedule
    def warmup_lambda(current_step):
        if current_step < train.warmup_steps:
            return float(current_step) / float(max(1, train.warmup_steps))
        return 1.0

    # Step-wise learning rate decay schedule
    def lr_decay_lambda(current_step):
        int_lr_decay_steps = [int(step * train.max_steps) for step in train.lr_decay_steps]
        if current_step in int_lr_decay_steps:
            return train.lr_decay_factor
        return 1.0

    # merge the two schedulers
    def combined_lambda(current_step):
        return warmup_lambda(current_step) * lr_decay_lambda(current_step)

    scheduler = LambdaLR(optimizer, lr_lambda=combined_lambda)

    return optimizer, scheduler
