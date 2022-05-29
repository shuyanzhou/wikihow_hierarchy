import numpy as np
import random
import torch
from transformers import AdamW
import os
from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def save_model(stats, model, optimizer, scheduler, args, fsave, **kwargs):
    save_info =  {
        'metric': stats,
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    save_info.update(kwargs)
    torch.save(save_info, fsave)
    print(f"save the model to {fsave}")

def restore(model, fpath, optimizer=None, scheduler=None):
    saved = torch.load(fpath)
    model.load_state_dict(saved["model"])
    if optimizer is not None:
        scheduler.load_state_dict(saved["scheduler"])
        optimizer.load_state_dict(saved["optim"])
    random.setstate(saved['system_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.random.set_rng_state(saved['torch_rng'])

def get_optimizer(model, args, total_steps):
    for name, param in model.named_parameters():
        param.requires_grad = True

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps=total_steps)
    return optimizer, scheduler

def init_distributed_mode(args):
    """
    Initialize distributed training parameters
        - local_rank
        - world_size
    This code snippet is adapted from fairseq.
    Copyright (c) Facebook, Inc. and its affiliates.
    """

    # if the process is launched using `torch.distributed.launch`
    if args.local_rank != -1:
        # read environment variables
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.n_gpu_per_node = int(os.environ['NGPU'])
    else:
        assert args.local_rank == -1
        args.local_rank = 0
        args.world_size = 1
        args.n_gpu_per_node = 1

    # define whether this is the master process / if we are in distributed mode
    args.is_master = args.local_rank == 0
    args.multi_gpu = args.world_size > 1

    # set GPU device
    if args.gpu:
        torch.cuda.set_device(args.local_rank)

    # initialize multi-GPU
    if args.multi_gpu:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function
        print(f"port: {os.environ['MASTER_PORT']}, "
              f"address: {os.environ['MASTER_ADDR']}, "
              f"world size: {os.environ['WORLD_SIZE']}, "
              f"rank: {os.environ['RANK']}")
        print("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
        )
