import os
import sys
from torch.utils.data import DataLoader
import torch
import json
from constants import logger, TQDM_DISABLE, WANDB_DISABLE
from train import Ranker, RerankDataset, CollateFunc, eval
import argparse

def inference(args):
    device = torch.device(f"cuda" if args.gpu else "cpu")
    logger.info(f'use device: {device}')

    args.device = device
    # model
    # restore
    saved = torch.load(args.model_path, map_location='cpu')
    train_args = saved['args']

    # overwrite some arguments
    train_args.neg_num = args.neg_num
    train_args.per_sample_tot = args.per_sample_tot
    train_args.max_sample = 10000000
    train_args.use_para_score = False if 'use_para_score' not in vars(train_args) else train_args.use_para_score
    train_args.null_token = "[unused2]" if 'null_token' not in vars(train_args) else train_args.null_token
    train_args.train_null = False if 'train_null' not in vars(train_args) else train_args.train_null
    train_args.gold_step_goal_para_score = './data/wikihow/gold.para.base.all.score' if 'gold_step_goal_para_score' not in vars(train_args) else train_args.gold_step_goal_para_score
    args.train_null = train_args.train_null
    args.null_token = train_args.null_token
    logger.info(f'number of candidates per sample: {train_args.per_sample_tot}')
    logger.info(f'train args: {vars(train_args)}')

    model = Ranker(train_args)
    model.load_state_dict(saved['model'])
    model.eval()
    model = model.to(device)

    logger.info(f"restore the model from {args.model_path}")

    test_ds = RerankDataset(train_args, args.test_path, pos_label=(not args.no_label))
    collate_fn = CollateFunc(train_args)
    test_data = DataLoader(dataset=test_ds, batch_size=args.bs, collate_fn=collate_fn, num_workers=4)
    metric, results = eval(test_data, model, args, return_prediction=True, pos_label=(not args.no_label))
    logger.info(metric)
    with open(args.save_path, "w+", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--model_path', default='')
    parser.add_argument('--test_path', default='')
    parser.add_argument('--save_path', default='')
    parser.add_argument('--no_label', help='the positive sample is not provided', action='store_true')
    parser.add_argument('--neg_num', type=int, default=29)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--mini_batch_size', type=int, default=30)
    args = parser.parse_args()
    args.per_sample_tot = args.neg_num + 1
    logger.info(vars(args))
    return args

if __name__ == "__main__":
    args = config()
    inference(args)
