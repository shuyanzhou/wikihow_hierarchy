import os
import sys
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import torch.nn.functional as F
import json
from transformers import BertConfig, BertTokenizer, BertModel, DebertaConfig, DebertaModel, DebertaTokenizer
from constants import logger, TQDM_DISABLE, WANDB_DISABLE
from utils import get_optimizer, save_model, restore
from tqdm import tqdm
import wandb
import argparse
import random
import numpy as np
import pickle
import copy
AVG_STEP_NUM = 6

MODEL_MAP = {'bert-base': 'bert-base-uncased', 'deberta': 'microsoft/deberta-large-mnli'}

class CollateFunc:
    def __init__(self, args):
        self.args = args
        if self.args.model_name == 'bert-base':
            self.tokenizer = BertTokenizer.from_pretrained(MODEL_MAP[self.args.model_name])
        elif self.args.model_name == 'deberta':
            self.tokenizer = DebertaTokenizer.from_pretrained(MODEL_MAP[self.args.model_name])
        else:
            raise NotImplementedError(self.args.model_name)

        self.tokenizer.pre_step_tok = '[unused0]'
        self.tokenizer.pre_step_tok_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pre_step_tok)
        self.tokenizer.post_step_tok = '[unused1]'
        self.tokenizer.post_step_tok_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.post_step_tok)

    def pad_batch(self, batch):
        text_list = []
        para_score_tensor = torch.zeros(self.args.per_sample_tot * len(batch))
        p_idx = 0
        for data in batch:
            step, gold, neg_list, step_index, cand_goal_para_score, corr_goal = data
            # add special token
            step[step_index] = f"{self.tokenizer.pre_step_tok} {step[step_index]} {self.tokenizer.post_step_tok}"
            context = '. '.join(step)
            cur_pos = f'{context} {self.tokenizer.sep_token} {gold}'
            text_list.append(cur_pos)
            para_score_tensor[p_idx] = cand_goal_para_score[gold]
            p_idx += 1
            for neg in neg_list:
                cur_neg = f'{context} {self.tokenizer.sep_token} {neg}'
                text_list.append(cur_neg)
                para_score_tensor[p_idx] = cand_goal_para_score[neg]
                p_idx += 1

        assert para_score_tensor.shape[0] == len(text_list)
        assert len(text_list) == self.args.per_sample_tot * len(batch)
        tok_text = self.tokenizer(text_list, return_tensors='pt', padding=True, truncation=False, add_special_tokens=True)

        return {'input': tok_text, 'para_score': para_score_tensor.float(), 'raw': batch}

    def __call__(self, batch):
        return self.pad_batch(batch)

class RerankDataset(Dataset):
    def __init__(self, args, data_file, pos_label=True):
        self.args = args
        self.data_file = data_file
        self.pos_label = pos_label
        logger.info(f"has label: {self.pos_label}")

        with open(self.args.step_goal_file, "r", encoding="utf-8") as f:
            self.step_goal = json.load(f)
            self.step_goal = {x['task']: x for x in self.step_goal}

        with open(self.args.step_goal_map_file, "r", encoding="utf-8") as f:
            self.step2goal = json.load(f)

        with open(self.args.gold_step_goal_para_score, "r", encoding="utf-8") as f:
            self.gold_sg_para_score = json.load(f)

        self.null_token = args.null_token
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def get_surr_context(self, step):
        cl = self.args.context_length
        try:
            step_context = self.step_goal[self.step2goal[step]]['caption']
        except KeyError:
            logger.info(f"{step} not found")
            return [step], 0
        _t = []
        for x in step_context:
            x = x if not x.endswith('.') else x[:-1].strip()
            _t.append(x)
        step_context = _t
        step_index = step_context.index(step)
        # get context
        pre = None
        post = None
        max_pre = step_index
        max_post = len(step_context) - step_index - 1
        if max_pre >= cl and max_post >= cl:
            pre = cl
            post = cl
        elif max_pre < cl:
            pre = step_index
            post = min(cl + cl - step_index, max_post)
        elif max_post < cl:
            post = len(step_context) - step_index - 1
            pre = min(max_pre, cl + cl - len(step_context) + step_index + 1)
        assert pre is not None and post is not None
        pre = int(pre)
        post = int(post)
        step_context = step_context[step_index - pre: step_index + post + 1]
        step_index = step_context.index(step)

        return step_context, step_index

    def construct_positive_negative_data(self, step, step_info, cand_goal_para_score):
        if self.pos_label:  # have label, for training and auto eval
            gold = step_info['gold_goal']
            neg = step_info['retrieved_goals'][:self.args.per_sample_tot]
            if gold in neg:
                neg.remove(gold)
            else:
                neg = neg[:-1]
            cand_goal_para_score[gold] = self.gold_sg_para_score[f"{step} || {gold}"]
        else:
            if len(step_info['retrieved_goals']) == self.args.per_sample_tot:
                gold = step_info['retrieved_goals'][0] # fake gold
                neg = step_info['retrieved_goals'][1:]
            else:  # sometime during retrieval, one candidate is removed
                gold = "x y z"  # dummy
                neg = step_info['retrieved_goals']
                neg += neg[:self.args.neg_num - len(neg)] # make it per_sample_num
        return gold, neg, cand_goal_para_score

    def construct_positive_negative_with_null(self, step, step_info, cand_goal_para_score):
        if self.pos_label:  # have label, for training and auto eval
            gold = step_info['gold_goal']
            neg = step_info['retrieved_goals'][:self.args.per_sample_tot]
            if gold in neg:
                neg.remove(gold)
                neg = neg[:-1]
                neg.append(self.null_token)
                cand_goal_para_score[gold] = self.gold_sg_para_score[f"{step} || {gold}"]
            else:
                neg = neg[:-1]
                gold = self.null_token
        else:
            if len(step_info['retrieved_goals']) == self.args.per_sample_tot:
                gold = step_info['retrieved_goals'][0] # fake gold
                neg = step_info['retrieved_goals'][1:-1]
                neg.append(self.null_token)
            else:  # sometime during retrieval, one candidate is removed
                gold = self.null_token  # dummy
                neg = step_info['retrieved_goals']
                neg += neg[:self.args.neg_num - len(neg)] # make it self.args.neg_num
        return gold, neg, cand_goal_para_score


    def load_data(self):
        data = []
        if ".json" in self.data_file:
            with open(self.data_file, "r", encoding="utf-8") as f:
                d = json.load(f)
        elif ".pkl" in self.data_file:
            with open(self.data_file, "rb") as f:
                d = pickle.load(f)
        else:
            raise NotImplementedError(self.data_file)
        # sanity check for step2goal
        miss = 0
        for step in d:
            if step not in self.step2goal:
                miss += 1
        logger.info(f"{miss} steps miss the goal in step2goal map")
        discard = 0

        # text
        for s_idx, (step, info) in enumerate(d.items()):
            cand_goal_para_score = {g: s for g, s in zip(info['retrieved_goals'], info['retrieved_goals_similarity'])}
            cand_goal_para_score[self.null_token] = min(info['retrieved_goals_similarity'])
            cand_goal_para_score['x y z'] = 0
            if not self.args.train_null:
                gold, neg, cand_goal_para_score = self.construct_positive_negative_data(step, info, cand_goal_para_score)
            else:
                gold, neg, cand_goal_para_score = self.construct_positive_negative_with_null(step, info, cand_goal_para_score)


            assert len(neg) == self.args.neg_num, (len(neg), step)
            if self.args.context_length != 0: # use context to encode each step
                step_context, step_index = self.get_surr_context(step)
            else:
                step_context = [step]
                step_index = 0

            if self.args.add_goal: # add title
                step_context = [self.step2goal.get(step, "goal")] + step_context
                step_index += 1

            # whole dataset
            if not self.pos_label and \
                    (max([len(' '.join(step_context + [x]).split()) for x in [gold] + neg]) >= 100 or max([len(' '.join(step_context + [x])) for x in [gold] + neg]) >= 600) :
                logger.info(f"discard {step_context}, {gold}, {max(neg, key=len)}")
                discard += 1
                continue


            if self.args.context_length >= 1 and max([len(' '.join(step_context + [x]).split()) for x in [gold] + neg]) >= 100:
                discard += 1
                continue

            corr_goal = None
            data.append([step_context, gold, neg, step_index, cand_goal_para_score, corr_goal])

        data = data[:self.args.max_sample]
        logger.info(f"load {len(data)} from {self.data_file}, discard {discard} samples")
        return data


    def load_step_context(self):
        pass

    def __getitem__(self, index):
        data = self.data[index]
        return data

## model
class Ranker(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # init a model
        if self.args.model_name == "bert-base":
            self.model_config = BertConfig()
            self.encoder = BertModel.from_pretrained(MODEL_MAP[self.args.model_name])
        elif self.args.model_name == "deberta":
            self.model_config = DebertaConfig(hidden_size=1024)
            self.encoder = DebertaModel.from_pretrained(MODEL_MAP[self.args.model_name])
        else:
            raise NotImplementedError

        self.class_num = 1
        self.dropout = nn.Dropout(self.args.hidden_dropout_prob)
        if self.args.use_para_score:
            logger.info("Use paraphrase score in the last layer as a feature")
            self.linear = nn.Linear(self.model_config.hidden_size + 1, self.class_num)
        else:
            self.linear = nn.Linear(self.model_config.hidden_size, self.class_num)

    def forward(self, inputs, para_score):
        embed = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs["attention_mask"], return_dict=True)
        if self.args.model_name == 'bert-base':
            tar_embed = embed.pooler_output
        elif self.args.model_name == 'deberta':
            tar_embed = embed.last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError(self.args.model_name)
        tar_embed = self.dropout(tar_embed)

        if self.args.use_para_score:
            para_score = para_score.unsqueeze(1)
            tar_embed = torch.cat([tar_embed, para_score], dim=1)

        logits = self.linear(tar_embed)
        logits = logits.reshape(-1, self.args.per_sample_tot)
        return logits

def calc_loss(pred, train_null, raw_data, null_token):
    # [bs * per_sample_tot, 1]
    bs = pred.shape[0]
    gold = torch.zeros((bs)).to(pred.device).long()
    ce = F.cross_entropy(pred, gold, reduction='sum')
    pred_label = torch.argmax(pred, dim=1)
    res_bool = pred_label == gold
    acc = torch.sum(res_bool).item()

    null_acc = 0
    null_tot = 1e-10
    if train_null: # calc null acc
        is_null = torch.zeros(bs)
        for idx, data in enumerate(raw_data):
            if data[1] == null_token:
                is_null[idx] = 1
        is_null = is_null.bool()
        null_acc = torch.sum(res_bool[is_null]).item()
        null_tot = torch.sum(is_null).item() + 1e-10


    return {'loss': ce, 'acc': acc, 'tot': bs, 'null_acc': null_acc, 'null_tot': null_tot}

def create_metric(loss, acc, tot, null_acc, null_tot):
    metric = {'loss': loss / tot, 'acc': acc / tot, 'null_acc': null_acc / null_tot}
    return metric

def update_ep_metric(ep_metric, bc_metric):
    for k, v in ep_metric.items():
        if isinstance(bc_metric[k], torch.Tensor):
            ep_metric[k] = v + bc_metric[k].item()
        else:
            ep_metric[k] = v + bc_metric[k]
    return ep_metric

def eval(data, model, args, return_prediction=False, pos_label=True):
    model.eval()
    torch.cuda.empty_cache()
    device = args.device
    ep_metric = {'loss': 0, 'tot': 1e-10, 'acc': 0, 'null_acc': 0, 'null_tot': 1e-10}
    bs_res = []
    for step, data in enumerate(tqdm(data, desc='eval', disable=TQDM_DISABLE)):
        try:
            assert args.bs == 1
            assert pos_label and (args.mini_batch_size == 30) or int(not pos_label)
            all_text = data['input']
            all_para_score = data['para_score']
            all_raw_text = data['raw']
            for ii in range(0, all_text['input_ids'].shape[0], args.mini_batch_size):
                cur_per_sample_tot = min(args.mini_batch_size, all_text['input_ids'].shape[0] - ii)
                text = copy.deepcopy(all_text)
                for kk, vv in all_text.items():
                    text[kk] = vv[ii:ii + cur_per_sample_tot]
                para_score = all_para_score[ii:ii + cur_per_sample_tot]
                # only support bs == 1
                raw_text = [all_raw_text[0]]
                text = text.to(device)
                para_score = para_score.to(device)
                model.args.per_sample_tot = cur_per_sample_tot
                # logger.info(caption['input_ids'].shape, data['other_data'])
                pred = model(text, para_score)
                bc_metric = calc_loss(pred, args.train_null, raw_text, args.null_token)
                bs = bc_metric['tot']
                update_ep_metric(ep_metric, bc_metric)

                if return_prediction:
                    pred_score = torch.softmax(pred, dim=1) if pos_label else pred
                    for i in range(bs):
                        cur_gold = raw_text[i][1] if pos_label else None
                        cur_res = {'step': raw_text[i][0][raw_text[i][3]].replace("[unused0]", "").replace("[unused1]", "").strip(),
                                   'gold': cur_gold, 'pred': {},
                                   'goal': None}
                        # gold + neg
                        cur_raw = [raw_text[i][1]] + raw_text[i][2]
                        cur_raw = cur_raw[ii:ii + cur_per_sample_tot]
                        for score, raw_goal in zip(pred_score[i], cur_raw):
                            cur_res['pred'][raw_goal] = score.item()
                        if len(bs_res) and bs_res[-1]['step'] == cur_res['step']:
                            bs_res[-1]['pred'] = {**bs_res[-1]['pred'], **cur_res['pred']}
                        else:
                            bs_res.append(cur_res)

                    # text id to text
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.info(f'| WARNING: ran out of memory, discard batch {step}')
                logger.info(data['raw'])
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

            # raise RuntimeError
    metric = create_metric(**ep_metric)
    if return_prediction:
        return metric, bs_res
    else:
        return metric

def run(args):
    device = torch.device(f"cuda" if args.gpu else "cpu")
    logger.info(f'use device: {device}, {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024}G')

    # model
    model = Ranker(args)
    model = model.to(device)
    args.device = device


    # dataset
    train_ds = RerankDataset(args, args.train_file)
    dev_ds = RerankDataset(args, args.dev_file)

    collate_fn = CollateFunc(args)

    train_data = DataLoader(dataset=train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate_fn, num_workers=8)
    dev_data = DataLoader(dataset=dev_ds, batch_size=args.val_bs, collate_fn=collate_fn, num_workers=4)

    # optimizer and scheduler
    # estimate the total steps
    tot_steps = len(train_data) // args.mega_bs * args.epochs
    logger.info(f"{tot_steps} steps in total")
    optimizer, scheduler = get_optimizer(model, args, tot_steps)
    if isinstance(args.resume, str) and len(args.resume):
        if args.cont_train:
            logger.info(f"reload the model, optimizer, scheduler on master process from {args.resume}")
            restore(model, args.resume, optimizer, scheduler)
        else:
            logger.info(f"reload the model on master process from {args.resume}, re-initialize optimizer and scheduler")
            restore(model, args.resume)
            model.args = args

    update_steps = 0
    for ep in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        ep_metric = {'loss': 0, 'tot': 1e-10, 'acc': 0, 'null_tot': 1e-10, 'null_acc': 0}
        for step, data in enumerate(tqdm(train_data, disable=TQDM_DISABLE)):
            try:
                text = data['input']
                para_score = data['para_score']
                text = text.to(device)
                para_score = para_score.to(device)

                # logger.info(caption['input_ids'].shape, data['other_data'])
                pred = model(text, para_score)
                bc_metric = calc_loss(pred, train_null=args.train_null, raw_data=data['raw'], null_token=args.null_token)
                cur_loss, bs =  bc_metric['loss'], bc_metric['tot']

                # backward
                cur_avg_loss = cur_loss / bs
                cur_avg_loss /= args.mega_bs
                cur_avg_loss.backward()

                cur_loss = cur_loss.item()

                if (step + 1) % args.mega_bs == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    update_steps += 1

                update_ep_metric(ep_metric, bc_metric)

                wandb.log({'train_loss': cur_loss / bs, 'train_acc': bc_metric['acc'] / bs,
                           'null_acc': bc_metric['null_acc'] / bc_metric['null_tot'],
                           'lr': scheduler.get_last_lr()[0]})

            except RuntimeError:
                print(data['raw'])
                raise RuntimeError

        train_metric = create_metric(**ep_metric)

        # eval
        logger.info('begin validation on master process ...')
        val_metric = eval(dev_data, model, args)
        # val_metric = {'loss': 0, 'acc': 0}

        # save model
        if ep >= args.min_save_ep:
            metric = {'train': train_metric, 'val': val_metric, 'epoch': ep}
            logger.info("begin saving the model on master process ...")
            save_model(metric, model, optimizer, scheduler, args, args.save_path.replace("epoch", f"ep{ep}"))

        # epoch log
        train_log = {f'train_{k}': v for k, v in train_metric.items()}
        val_log = {f'val_{k}': v for k, v in val_metric.items()}
        wandb.log({**train_log, **{'epoch': ep}})
        wandb.log({**val_log, **{'epoch': ep}})
        logger.info(f"train epoch {ep}, {train_metric}")
        logger.info(f"dev epoch {ep}, {val_metric}")

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', default=True)
    # files
    parser.add_argument('--step_goal_file', default='./data/wikihow.json')
    parser.add_argument('--step_goal_map_file', default='./data/step2goal.json')
    parser.add_argument('--model_name', default='bert-base')
    parser.add_argument('--train_file', default='')
    parser.add_argument('--dev_file', default='')
    parser.add_argument('--save_path', default='')
    parser.add_argument('--gold_step_goal_para_score', help='file that store the retrieval score',
                        default='')

    # hyper parameters
    parser.add_argument('--resume', default="")
    parser.add_argument('--cont_train', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--val_bs', type=int, default=2)
    parser.add_argument('--mega_bs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--min_save_ep", type=int, default=2)

    # data specific
    parser.add_argument("--neg_num", help='number of negative each sample', type=int, default=9)
    parser.add_argument("--context_length", help="how much context to use for each step", type=int, default=0)
    parser.add_argument("--add_goal", help='whether to add the goal to the beginning', action='store_true')
    parser.add_argument("--use_para_score", help="whether to use paraphrase score as a feature before MLP", action='store_true')
    parser.add_argument("--train_null", help="train with a dummy node for non-existent goal", action='store_true')

    # debug
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max_sample', type=int, default=10000000)

    args = parser.parse_args()
    args.per_sample_tot = args.neg_num + 1 # number of sentences per sample
    args.null_token = "[unused2]"

    assert "epoch" in args.save_path, (args.save_path)
    if args.debug:
        logger.info(f"[WARNING] debug mode is on")
        args.max_sample = 50
        args.epochs = 100
        args.min_save_ep=1000
    logger.info(vars(args))
    wandb.config.update(args)
    return args

if __name__ == "__main__":
    if WANDB_DISABLE:
        wandb.init(mode='disabled')
    else:
        wandb.init(project='wikihow_rerank')

    SEED = 326
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    args = config()
    run(args)