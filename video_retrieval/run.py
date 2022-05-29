import os
import sys
import torch.multiprocessing as mp
import json
import pickle
from tqdm import tqdm
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import elasticsearch
from collections import defaultdict, Counter
import argparse
import numpy as np
from sklearn.metrics import average_precision_score
import copy


DEBUG = False
FROM_SCRATCH = False
HOST='' # set your own elasticsearch server
def preprocess():
    with open("./data/howto100m/video1k.test.caption.json", "r") as f:
        d = json.load(f)
        _d = {}
        for k, v in d.items():
            end = v[-1]['start'] + v[-1]['duration']
            _v = [x['text'] for x in v if x['start'] >= 10 and x['start'] + x['duration'] <= end - 10]
            _v = ' '.join(_v)
            _d[k] = _v
    print(len(_d))
    with open("./data/howto100m/video1k.test.caption.cat.json", "w+") as f:
        json.dump(_d, f, indent=2)


class SESearch:
    def __init__(self, caption_file_list: list, index: str):
        self.es = Elasticsearch(timeout=60, host=HOST)
        self.caption_file_list = caption_file_list
        self.index = index
        if FROM_SCRATCH:
            x = input("ARE YOU SURE FROM SCRATCH? ")
            if x.lower() in ['yes', 'y']:
                self.es.indices.delete(index=self.index, ignore=[400, 404])
                print(f"delete {self.index}")
                self.es.indices.create(index=self.index)
                print(self.es.indices.get_alias().keys())
                all_docs = list(self.gendata())
            else:
                print("886")
                exit(0)
        print(f"done init the index {self.index}")
        self.es.indices.refresh(self.index)
        print(self.es.cat.count(self.index, params={"format": "json"}))

    def gendata(self):
        for caption_file in self.caption_file_list:
            with open(caption_file, "r") as f:
                cap_d = json.load(f)
                for vid, cap in cap_d.items():
                    result = {
                        "_index": self.index,
                        "_type": "_doc",
                        'source_content': ' '.join(cap) if isinstance(cap, list) else cap,
                        'video_id': vid
                    }
                    yield result

    def create_index(self):
        all_docs = list(self.gendata())
        print(bulk(self.es, all_docs, index=self.index))

def calc_mean_rank(rank_counter, print_flag=True):
    tot = sum(rank_counter.values())
    mr = 0
    for k, v in rank_counter.items():
        mr += k * v
    mr /= tot
    if print_flag:
        print(f"number of videos in total: {tot}")
    return mr

def calc_mrr(rank_counter, print_flag=True):
    tot = sum(rank_counter.values())
    mrr = 0
    for k, v in rank_counter.items():
        mrr += 1 / k * v
    mrr /= tot
    if print_flag:
        print(f"number of videos in total: {tot}")
    return mrr

def calc_average_precision(retrieved, r_size):
    gold_rank = list(retrieved['gold'].values())
    # create a fake score
    y_true = [0 for _ in range(r_size)]
    for r in gold_rank:
        if r <= r_size - 1:
            y_true[r] = 1
    if sum(y_true) == 0:
        return 0
    else:
        y_scores = [1 / (i + 1) for i in range(r_size)]
        aps = average_precision_score(y_true, y_scores)
        # the right relevant number
        aps = aps * sum(y_true) / len(gold_rank)
        return aps

def calc_mean_average_precision(retrieved, r_size):
    map = []
    for goal, info in retrieved.items():
        ap = calc_average_precision(info, r_size)
        map.append(ap)
    return np.mean(map)

class CESearch:
    def __init__(self, args):
        self.args = args
        self.searcher = SESearch(args.caption_file, self.args.index)
        self.index = self.args.index
        self.r_size = 200
        with open(args.meta_file, "r") as f:
            d = json.load(f)
            self.vid_task_map = d['vid_task_map']

        cap_exist = set()
        for caption_file in args.caption_file:
            with open(caption_file, "r") as f:
                cap_exist = cap_exist.union(set(json.load(f).keys()))

        with open(args.video_length_file, "r") as f:
            self.video_len_map = json.load(f)

        with open(args.test_file, "r") as f:
            test_data = json.load(f)
            tot_v = 0
            # all
            _test_data = {}
            for task in test_data['train'].keys():
                _test_data[task] = []
                for split in self.args.allow_split:
                    _test_data[task] += test_data[split][task]

            self.test_data = _test_data
            for k, v in self.test_data.items():
                v = [x for x in v if x in cap_exist and self.video_len_map[x] >= self.args.min_video_len]
                tot_v += len(v)
                self.test_data[k] = v
            # remove empty goals
            self.test_data = {k: v for k, v in self.test_data.items() if len(v)}



    def get_query(self):
        with open(args.base_query_file, "r") as f:
            self.base_query_map = json.load(f)
            self.base_query_map = {x['task']: {'goal': x['task'], 'step': x['caption'], 'exp_step': ''} for x in self.base_query_map}

        with open(args.query_file, "r") as f:
            self.query_map = json.load(f)
            _query_map = {}
            for item in self.query_map:
                base_step = [x if not x.endswith(".") else x[:-1].strip() for x in self.base_query_map[item['task']]['step']]
                _query_map[item['task']] = {'goal': item['task'], 'step': ' || '.join(base_step),
                                            'exp_step':' || '.join([x for x in item['caption'] if x not in base_step])}
        self.query_map = _query_map

    def get_topk(self, query: dict, field: str, r_size):
        if not self.args.weight_query:
            raise NotImplementedError
        else:
            goal_weight = self.args.goal_weight
            step_weight = self.args.step_weight
            exp_step_weight = self.args.exp_step_weight if len(query['exp_step'].strip()) != 0 else 0
            weight_q = {
                      "query": {
                        "bool": {
                          "should": [
                            {
                              "function_score": {
                                "query": {
                                  "match": {
                                    field: query['goal']
                                  }},
                                "boost": goal_weight}
                            },
                            {
                              "function_score": {
                                  "query": {
                                      "match": {
                                          field: query['exp_step'].replace(" || ", " ")
                                      }},
                                  "boost": exp_step_weight}
                            },
                            {
                              "function_score": {
                                  "query": {
                                      "match": {
                                          field: query['step'].replace(" || ", " ")
                                      }},
                                  "boost": step_weight}
                            },
                          ]}}, 'size': r_size}
            results = self.searcher.es.search(
                index=self.index,
                body=weight_q)


        return results['hits']['hits']

    def retrieve_video(self, goal, video_list, cur_query, retrieved, recall_n, precision_n, rank_counter, r_size):
        try:
            results = self.get_topk(cur_query, field=self.args.source_field, r_size=r_size)
        except elasticsearch.exceptions.RequestError as e:
            # print(f"skip {goal}")
            assert "400" in repr(e)
            print("\n", repr(e))
            return

        results_vid = [x['_source']['video_id'] for x in results]
        results_score = [x['_score'] for x in results]
        retrieved[goal] = {'gold': None,
                           'top-50': [f"{x} || {self.vid_task_map[x]} || {score}" for x, score in
                                      zip(results_vid[:50], results_score)]}

        for tk in recall_n.keys():
            cur_result_vids = results_vid[:tk]
            cur_hit = sum([x in cur_result_vids for x in video_list])
            recall_n[tk] = recall_n[tk] + cur_hit / len(video_list)
            precision_n[tk] = precision_n[tk] + cur_hit / tk

        gold_rank = {}
        for v in video_list:
            try:
                rank_counter[results_vid.index(v) + 1] += 1
                gold_rank[v] = results_vid.index(v) + 1
            except ValueError:
                rank_counter[r_size + 1] += 1
                gold_rank[v] = r_size + 1
        retrieved[goal]['gold'] = gold_rank

    def calc_median(self, rank_counter):
        rank_list = []
        for k, v in rank_counter.items():
            rank_list += [int(k) for _ in range(int(v))]
        med_r = np.median(rank_list)
        return med_r

    def print_result(self, test_data, recall_n, precision_n, rank_counter, r_size):
        recall_n = {k: v / len(test_data) for k, v in recall_n.items()}
        precision_n = {k: v / len(test_data) for k, v in precision_n.items()}

        # mean rank
        for k in sorted(recall_n.keys()):
            print(f"{recall_n[k] :.3f}", end="\t")
        print()
        for k in sorted(precision_n.keys()):
            print(f"{precision_n[k] :.3f}", end="\t")
        print()
        for k in sorted(recall_n.keys()):
            print(f"{2 * precision_n[k] * recall_n[k] / (precision_n[k] + recall_n[k]) :.3f}", end="\t")
        print()
        mr = calc_mean_rank(rank_counter)
        print(f"{mr :.3f}")

        med_r = self.calc_median(rank_counter)
        print(f"{med_r :.3f}")


    def eval_dataset(self, test_data):
        print(f"number of goals in total: {len(test_data)}")
        r_size = self.r_size
        rank_counter = Counter()
        retrieved = {}
        recall_n = {1: 0, 3: 0, 5: 0, 10: 0, 15: 0, 25: 0, 50: 0, 100: 0, 150: 0, 200: 0, 300: 0, 400: 0, 500: 0}
        precision_n = {1: 0, 3: 0, 5: 0, 10: 0, 15: 0, 25: 0, 50: 0, 100: 0, 150: 0, 200: 0, 300: 0, 400: 0, 500: 0}

        for goal, video_list in tqdm(test_data.items(), disable=False):
            if len(video_list) == 0:
                continue
            cur_query = self.query_map[goal]
            self.retrieve_video(goal, video_list, cur_query, retrieved, recall_n, precision_n, rank_counter, r_size)
        self.print_result(test_data, recall_n, precision_n, rank_counter, r_size)
        return retrieved

    def single_query_measure(self, goal, video_list, cur_query, r_size):
        rank_counter = Counter()
        retrieved = {}
        recall_n = {1: 0, 3: 0, 5: 0, 10: 0, 15: 0, 25: 0, 50: 0, 100: 0}
        precision_n = {1: 0, 3: 0, 5: 0, 10: 0, 15: 0, 25: 0, 50: 0, 100: 0}
        self.retrieve_video(goal, video_list, cur_query, retrieved, recall_n, precision_n, rank_counter, r_size)
        return rank_counter, recall_n, precision_n, retrieved

    @staticmethod
    def gen_next_state(state):
        for idx, selected in enumerate(state):
            if selected == 0:
                next_state = copy.deepcopy(state)
                next_state[idx] = 1
                yield next_state

    def search_step_comb(self, test_data, base_only, max_step, penalty):
        print(f"number of goals in total: {len(test_data)}")
        r_size = self.r_size
        helpful_step_map = {}
        helpful_step_info = {}
        if 'mrr' in self.args.help_version:
            cost_func = lambda x: -calc_mrr(x[0], print_flag=False)
            print(f"mrr as cost")
        elif 'ap' in self.args.help_version:
            cost_func = lambda x: -calc_average_precision(x[3][list(x[3].keys())[0]], self.r_size)
            print(f"average precision as cost")
        elif 'mr' in self.args.help_version:
            cost_func = lambda x: calc_mean_rank(x[0], print_flag=False)
            print(f"mean rank as cost")
        else:
            raise NotImplementedError(args.help_version)

        for _, (goal, video_list) in enumerate(tqdm(test_data.items(), disable=True)):
            helpful_step_info[goal] = {}
            print("===========================")
            print(goal)
            if len(video_list) == 0:
                continue
            cur_query = self.query_map[goal]
            base_steps = cur_query['step'].split(" || ")
            exp_steps = cur_query['exp_step'].split(" || ")
            helpful_steps = []
            if base_only:
                candid_steps = base_steps
            else:
                candid_steps = base_steps + exp_steps

            # init [0, 0, .. for all]
            cur_query['step'] = ''
            cur_query['exp_step'] = ''
            r_result = self.single_query_measure(goal, video_list, cur_query, r_size)
            base_cost = cost_func(r_result)

            min_cost = base_cost
            best_state = [0 for _ in candid_steps]

            for _ in range(min(len(candid_steps), max_step)):
                best_in_state = None
                min_in_cost = 1000
                for next_state in self.gen_next_state(best_state):
                    selected_steps = [candid_steps[idx] for idx, selected in enumerate(next_state) if selected == 1]
                    selected_steps = ' || '.join(selected_steps)
                    cur_query['step'] = selected_steps
                    r_result = self.single_query_measure(goal, video_list, cur_query, r_size)
                    # cur_cost = calc_mean_rank(rank_counter, print_flag=False)
                    cur_cost = cost_func(r_result)
                    if cur_cost < min_in_cost:
                        best_in_state = copy.deepcopy(next_state)
                        min_in_cost = cur_cost
                if min_in_cost + penalty < min_cost:
                    best_state = best_in_state
                    min_cost = min_in_cost
                else:
                    break

            selected_steps = [candid_steps[idx] for idx, selected in enumerate(best_state) if selected == 1]
            for s in selected_steps:
                if s in base_steps:
                    helpful_steps.append([s, '[org]'])
                else:
                    helpful_steps.append([s, '[exp]'])
            helpful_step_map[goal] = helpful_steps
            helpful_step_info[goal] = [base_cost, min_cost]
            print(helpful_steps)
            print(base_cost, min_cost)

        helpful = {'step_map': helpful_step_map, 'info': helpful_step_info}
        return helpful


    def check_each_step(self, test_data):
        print(f"number of goals in total: {len(test_data)}")
        r_size = 200
        helpful_step_map = {}
        helpful_step_info = {}
        for _, (goal, video_list) in enumerate(tqdm(test_data.items(), disable=True)):
            helpful_step_info[goal] = {}
            print("===========================")
            print(goal)
            if len(video_list) == 0:
                continue
            cur_query = self.query_map[goal]
            base_steps = cur_query['step'].split(" || ")
            exp_steps = cur_query['exp_step'].split(" || ")
            helpful_steps = []
            steps_info = {}
            for step_idx, step in enumerate([''] + base_steps + exp_steps):
                # per query measure
                rank_counter = Counter()
                retrieved = {}
                recall_n = {1: 0, 3: 0, 5: 0, 10: 0, 15: 0, 25: 0, 50: 0, 100: 0}
                precision_n = {1: 0, 3: 0, 5: 0, 10: 0, 15: 0, 25: 0, 50: 0, 100: 0}

                cur_query['step'] = step
                cur_query['exp_step'] = ''
                self.retrieve_video(goal, video_list, cur_query, retrieved, recall_n, precision_n, rank_counter, r_size)

                tar_recall = 25
                if step_idx == 0:
                    base_avg_rank = calc_mean_rank(rank_counter, print_flag=False)
                    base_result = retrieved[goal]['gold']
                    base_recall = recall_n[tar_recall]
                    steps_info['goal'] = {'rank': base_avg_rank, 'recall': base_recall}
                    print("goal only", f"mean rank: {base_avg_rank :.2f}, recall: {base_recall}")
                else:
                    assert base_avg_rank is not None
                    assert base_recall is not None
                    exp_avg_rank = calc_mean_rank(rank_counter, print_flag=False)
                    exp_result = retrieved[goal]['gold']
                    exp_recall = recall_n[tar_recall]
                    assert len(exp_result) == len(base_result)
                    # check how many steps are matched
                    win_num = sum([int(exp_result[x] < base_result[x]) for x in base_result])
                    loss_num = sum([int(exp_result[x] > base_result[x]) for x in base_result])
                    if exp_avg_rank < base_avg_rank or exp_recall > base_recall:
                    # if base_recall < exp_recall:
                        if step in exp_steps:
                            mark = "[exp]"
                        else:
                            mark = "[org]"
                        helpful_steps.append([step, mark])
                        steps_info[step] = {'rank': exp_avg_rank, 'recall': exp_recall, 'win': win_num, 'loss': loss_num, 'source': mark}
                        print(mark, step, f"mean rank {exp_avg_rank :.2f}, recall: {exp_recall :.2f}, win: {win_num}, loss: {loss_num}")
            helpful_step_map[goal] = helpful_steps
            helpful_step_info[goal] = steps_info
        helpful = {'step_map': helpful_step_map, 'info': helpful_step_info}
        return helpful

def prune_helpful_map(helpful_map, helpful_info, rank_key, topn=1, keep_only_one=False):
    exp = 0
    org = 0
    for goal, v in helpful_info.items():
        if len(helpful_info[goal]) == 1:
            continue
        if rank_key == 'rank':
            reverse = False
        elif rank_key in ['recall', 'win']:
            reverse = True
        else:
            raise NotImplementedError
        _helpful_info = copy.deepcopy(helpful_info[goal])
        _helpful_info.pop('goal')
        most_helpful_step = sorted(_helpful_info.items(), key=lambda x: x[1][rank_key], reverse=reverse)
        most_helpful_step = [x for x in most_helpful_step if x[1]['rank'] < helpful_info[goal]['goal']['rank']]
        # most_helpful_step = [x for x in most_helpful_step if x[1]['loss'] < x[1]['win']]

        # most_helpful_step = [x for x in most_helpful_step if x[1]['rank'] < helpful_info[goal]['goal']['rank'] and x[1]['recall'] > helpful_info[goal]['goal']['recall']]
        # if reverse:
        #     most_helpful_step = [x for x in most_helpful_step if x[1][rank_key] > helpful_info[goal]['goal'][rank_key]]
        # else:
        #     most_helpful_step = [x for x in most_helpful_step if x[1][rank_key] < helpful_info[goal]['goal'][rank_key]]
        most_helpful_org = [x for x in most_helpful_step if x[1].get('source', None) == '[org]'][:topn]
        most_helpful_exp = [x for x in most_helpful_step if x[1].get('source', None) == '[exp]'][:topn]
        if topn == 1 and keep_only_one and len(most_helpful_org) and len(most_helpful_exp):
            if most_helpful_org[0][1]['rank'] > most_helpful_exp[0][1]['rank']:
                most_helpful_org = []
            elif most_helpful_org[0][1]['rank'] < most_helpful_exp[0][1]['rank']:
                most_helpful_exp = []
        helpful_map[goal] = [[x[0], x[1]['source']] for x in most_helpful_org + most_helpful_exp]
        org += sum([x[1] == '[org]' for x in helpful_map[goal]])
        exp += sum([x[1] == '[exp]' for x in helpful_map[goal]])
    print(f"original: {org}, expansion: {exp}")


def save_result(d, file_name):
    with open(file_name, "w+") as f:
        json.dump(d, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_field', choices=('source_content', 'title'), default='source_content')
    parser.add_argument('--caption_file', nargs='+', default=['./data/video_retrieval/video1k.train.caption.json',
                                                              './data/video_retrieval/video1k.dev.caption.json',
                                                              './data/video_retrieval/video1k.test.caption.json'])

    parser.add_argument('--base_query_file', default='./data/video_retrieval/step_goal.para.d0.all_base.howto1k.all.json')
    parser.add_argument('--video_length_file', default="./data/video_retrieval/video_length.json")
    parser.add_argument('--query_file', default="")
    parser.add_argument('--test_file', default="./data/video_retrieval/split_task_video_map.json")
    parser.add_argument('--allow_split', nargs='+', default=['train', 'dev', 'test'])
    parser.add_argument('--goal_only', action='store_true')
    parser.add_argument('--concat_goal', action='store_true')
    parser.add_argument('--meta_file', default='./data/howto100m/video1k/video1k.meta_map.json')
    parser.add_argument('--weight_query', action='store_true', default=True)
    parser.add_argument('--goal_weight', type=float)
    parser.add_argument('--step_weight', type=float)
    parser.add_argument('--exp_step_weight', type=float)
    parser.add_argument('--min_video_len', type=float, default=0)
    parser.add_argument('--mode', type=float, default=2)
    parser.add_argument('--test_mode', nargs='+', default=[5.1, 5.2])
    parser.add_argument('--help_version', default='mr')
    parser.add_argument('--helpful_file', default="./data/video_retrieval/all_goal_useful_step.[help_version].json")
    parser.add_argument('--max_climb_step', type=int, default=15)
    parser.add_argument('--penalty', type=float, default=0)
    parser.add_argument('--goal_list', default="./data/video_retrieval/goal_list.json")
    parser.add_argument('--query_file_idx', type=int, default=4)
    args = parser.parse_args()

    query_list = ['step_goal.para.d0.sample_base.howto1k.all.json',
                  'step_goal.para.d0.sample_base.howto1k.all.json',
                  'step_goal.para.d1.all_base.all_expansion.rerank.para_score.90.howto1k.all.json',
                  'step_goal.para.d1.all_base.all_expansion.rerank.train_null00.50.howto1k.all.json',
                  'step_goal.para.d1.all_base.all_expansion.rerank.goal.c1.train_null00.50.howto1k.all.json',
                  'step_goal.para.d1.all_base.all_expansion.rerank.goal.c1.para_score90.howto1k.all.json']
    # 1 for train, 2 for dev 3 for test
    # 1.1 comb base 1.2 comb exp
    args.test_mode = [float(x) for x in args.test_mode]
    args.query_file = f"./data/wikihow/{query_list[args.query_file_idx]}"

    print(vars(args))
    if args.mode == 0:
        FROM_SCRATCH = True
        args.allow_split =   ['train']
        args.index = 'video_1k_t150_resplit_train'
        args.caption_file = ['./data/video_retrieval/video1k.train.caption.json']
        search = CESearch(args)

        args.allow_split = ['dev']
        args.index = 'video_1k_t150_resplit_dev'
        args.caption_file = ['./data/video_retrieval/video1k.dev.caption.json']
        search = CESearch(args)

        args.allow_split = ['test']
        args.index = 'video_1k_t150_resplit_test'
        args.caption_file = ['./data/video_retrieval/video1k.test.caption.json']
        search = CESearch(args)

    # elif args.mode == 1:
    #     with open(args.goal_list, "r") as f:
    #         train_goal = json.load(f)
    #     args.goal_weight = 1.0
    #     args.step_weight = 1.0
    #     args.allow_split = ['train']
    #     args.index = 'video_1k_t150_resplit_train'
    #     search = CESearch(args)
    #     search.get_query()
    #     test_data = {k: v for idx, (k, v) in enumerate(search.test_data.items()) if k in train_goal}
    #     helpful_map = search.check_each_step(test_data)
    #     helpful_file = args.helpful_file.replace("[help_version]", f"{args.help_version}")
    #     with open(helpful_file, "w+") as f:
    #         json.dump(helpful_map, f, indent=2)
    #     print(f"save helpful map to {helpful_file}")

    elif args.mode == 1.1:
        with open(args.goal_list, "r") as f:
            train_goal = json.load(f)

        args.goal_weight = 1.0
        args.step_weight = 1.0
        args.allow_split = ['train']
        args.index = 'video_1k_t150_resplit_train'
        search = CESearch(args)
        search.get_query()
        test_data = {k: v for idx, (k, v) in enumerate(search.test_data.items()) if k in train_goal}
        helpful_map = search.search_step_comb(test_data, base_only=True, max_step=args.max_climb_step, penalty=args.penalty)
        helpful_file = args.helpful_file.replace("[help_version]", f"{args.help_version}.base.m{args.max_climb_step}.{args.penalty}")
        with open(helpful_file, "w+") as f:
            json.dump(helpful_map, f, indent=2)
        print(f"save helpful map to {helpful_file}")

    elif args.mode == 1.2:
        # exp comb
        with open(args.goal_list, "r") as f:
            train_goal = json.load(f)

        args.goal_weight = 1.0
        args.step_weight = 1.0
        args.allow_split = ['train']
        args.index = 'video_1k_t150_resplit_train'
        search = CESearch(args)
        search.get_query()
        test_data = {k: v for idx, (k, v) in enumerate(search.test_data.items()) if k in train_goal}
        helpful_map = search.search_step_comb(test_data, base_only=False, max_step=args.max_climb_step, penalty=args.penalty)
        helpful_file = args.helpful_file.replace("[help_version]", f"{args.help_version}.exp.m{args.max_climb_step}.{args.penalty}")
        with open(helpful_file, "w+") as f:
            json.dump(helpful_map, f, indent=2)
        print(f"save helpful map to {helpful_file}")

    elif args.mode == 2:
        # hyper
        rank_key = 'rank'
        keep_only_one = False
        prune_topn = 1

        args.goal_weight = 1.0
        args.allow_split = ['train']
        args.index = 'video_1k_t150_resplit_train'
        # args.step_weight = 1
        # args.exp_step_weight = 0
        # args.allow_split = ['dev']
        # args.index = 'video_1k_t150_resplit_dev'
        # args.allow_split = ['test']
        # args.index = 'video_1k_t150_resplit_test'

        with open(args.goal_list, "r") as f:
            eval_goal = json.load(f)

        if 1 in args.test_mode:
            print("\t\t=========================goal only=========================")
            args.step_weight = 0.0
            args.exp_step_weight = 0.0
            print(args.goal_weight, args.step_weight, args.exp_step_weight)
            search = CESearch(args)
            search.get_query()
            test_data = {k: v for idx, (k, v) in enumerate(search.test_data.items()) if k in eval_goal}
            result = search.eval_dataset(test_data)
            save_result(result, f"./data/video_retrieval/retrieved.goal_only.{args.index}.json")

        if 2 in args.test_mode:
            print("\t\t=========================all steps=========================")
            args.step_weight = 0.1
            args.exp_step_weight = 0.0
            print(args.goal_weight, args.step_weight, args.exp_step_weight)
            search = CESearch(args)
            search.get_query()
            test_data = {k: v for idx, (k, v) in enumerate(search.test_data.items()) if k in eval_goal}
            result = search.eval_dataset(test_data)
            save_result(result, f"./data/video_retrieval/retrieved.all_step.{args.index}.{args.step_weight}.json")


        if 5.1 in args.test_mode:
            helpful_file = args.helpful_file.replace("[help_version]", f"{args.help_version}.base.m{args.max_climb_step}.{args.penalty}")
            print(f"load helpful map from {helpful_file}")
            with open(helpful_file, "r") as f:
                helpful = json.load(f)
                helpful_map = helpful['step_map']
                helpful_info = helpful['info']

            print("\t\t==========================combined base steps=========================")
            # for sw in np.arange(0.4, 0.5, 0.1):
            #     args.step_weight = sw
            #     args.exp_step_weight = 0
            args.step_weight = 0.6
            args.exp_step_weight = 0
            print(args.goal_weight, args.step_weight, args.exp_step_weight)
            search = CESearch(args)
            search.get_query()
            for k, v in search.query_map.items():
                v['step'] = ' || '.join([x[0] for x in helpful_map[k]])
                v['exp_step'] = ''
            test_data = {k: v for idx, (k, v) in enumerate(search.test_data.items()) if k in eval_goal}
            result = search.eval_dataset(test_data)
            save_result(result,f"./data/video_retrieval/retrieved.filter.base.exp.{args.help_version}.base.{args.penalty}.{args.index}.{args.step_weight}.{args.exp_step_weight}.json")

        if 5.2 in args.test_mode:
            helpful_file = args.helpful_file.replace("[help_version]", f"{args.help_version}.exp.m{args.max_climb_step}.{args.penalty}")
            print(f"load helpful map from {helpful_file}")
            with open(helpful_file, "r") as f:
                helpful = json.load(f)
                helpful_map = helpful['step_map']
                helpful_info = helpful['info']

            print("\t\t==========================combined base+exp steps=========================")
            args.step_weight = 1.0
            args.exp_step_weight = 0
            print(args.goal_weight, args.step_weight, args.exp_step_weight)
            search = CESearch(args)
            search.get_query()
            for k, v in search.query_map.items():
                v['step'] = ' || '.join([x[0] for x in helpful_map[k]])
                v['exp_step'] = ''
            test_data = {k: v for idx, (k, v) in enumerate(search.test_data.items()) if k in eval_goal}
            result = search.eval_dataset(test_data)
            save_result(result,f"./data/video_retrieval/retrieved.filter.base.exp.{args.help_version}.exp.{args.penalty}.{args.index}.{args.step_weight}.{args.exp_step_weight}.json")




