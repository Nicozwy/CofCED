import re
import numpy as np
from rouge_score import rouge_scorer, scoring
from multiprocessing import Pool
from collections import Counter

import os
import json

score_names = ['rouge1', 'rouge2', 'rougeLsum']

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def compute_rouge_from_array(decoded: list, references: list):
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)
    with Pool() as p:

        # scores = p.starmap(scorer.score, [(p, t) for p, t in zip(decoded, references)])
        scores = p.starmap(scorer.score, [(t, p) for t, p in zip(references, decoded)])

        aggregator = scoring.BootstrapAggregator()

        for score in scores:
            aggregator.add_scores(score)
        aggregated_scores = aggregator.aggregate()

        result_dict = {}
        for score_name in score_names:
            mid = aggregated_scores[score_name].mid
            # low = aggregated_scores[score_name].low
            # high = aggregated_scores[score_name].high

            result_dict[f'{score_name}_precision'] = mid.precision
            result_dict[f'{score_name}_recall'] = mid.recall
            result_dict[f'{score_name}_f1'] = mid.fmeasure

    return result_dict


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def rouge_results_to_str(results_dict):
    keys = ['rouge1_recall', 'rouge1_precision', 'rouge1_f1',
            'rouge2_recall', 'rouge2_precision', 'rouge2_f1',
            'rougeLsum_recall', 'rougeLsum_precision', 'rougeLsum_f1']

    print('\t'.join(keys))
    print('\t'.join([str(results_dict[_k]) for _k in keys]))

    return ">> ROUGE-F(1/2/l): {:.2f} & {:.2f} & {:.2f}\nROUGE-R(1/2/l): {:.2f} & {:.2f} & {:.2f}\nROUGE-P(1/2/l): {:.2f} & {:.2f} & {:.2f}\n".format(
        results_dict["rouge1_f1"]*100, results_dict["rouge2_f1"]*100,
        results_dict["rougeLsum_f1"]*100, results_dict["rouge1_recall"]*100,
        results_dict["rouge2_recall"]*100, results_dict["rougeLsum_recall"]*100,
        results_dict["rouge1_precision"]*100, results_dict["rouge2_precision"]*100, results_dict["rougeLsum_precision"]*100)



## oracle rouge 
if __name__ == '__main__':
    print("cla... oracle rouge score")
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/dataset/LIAR-RAW/"
    
    justification = []#gold
    oracles = []
    lead = []
    label_dis = []

    total_claims = []
    # 1 读取 explain 和 oracle sentences: reports , tokenized, {sent, is_evidence}
    for name in ['test', 'val', 'train']:
    # for name in ['train']:
        ds_path = path + name + '.json'   
        with open(ds_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
            print(len(all_data))
            total_claims.append(len(all_data))
            for data in all_data:
                label_dis.append(data['label'])
                justification.append(data['explain'])
                tok_sents = []
                N = 4
                for report in data['reports']:
                    for tok in report['tokenized']:
                        if N>0:
                            N -= 1
                            lead.append(tok['sent'])
                        if tok['is_evidence'] == 1:
                            tok_sents.append(tok['sent'])
                
                oracles.append('\n '.join(tok_sents))
                

        # 2 调用compute_rouge_from_array()
        # rouges = compute_rouge_from_array(oracles, justification) 
        # print(f'{name} justification Rouges = \n%s' % (rouge_results_to_str(rouges)))

        # rouges = compute_rouge_from_array(oracles, lead) 
        # print(f'{name} lead Rouges = \n%s' % (rouge_results_to_str(rouges)))

    print('total claim nums: ', sum(total_claims))
    print(Counter(label_dis))






    