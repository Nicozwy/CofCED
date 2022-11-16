# coding: utf-8
from statistics import mean 
from datetime import datetime
from logging import log
import os
from os.path import join as pjoin
from posixpath import join
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
# from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from helpers.reader5 import myDataset
# from dataset import End2EndDataset
from helpers.torch_util import calc_f1
from helpers.path_util import from_project_root

from helpers.simple_logger import SimpleLogger
from helpers.utils import rouge_results_to_str, compute_rouge_from_array

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import sys
sys.path.append('helpers/')
sys.path.append('model/')
from tqdm import tqdm
# from logger import Logger
# from model import BasicFC

# TOP_K = 12#55 
TOP_K = 12#55
# MAX_ORACLE= 55 # 最多的oracle的句子数
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# ROOT_PROJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "AFC/dataset/oracles"
LABEL_IDS = {"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5} #LIAR & LIAR-RAW
# LABEL_IDS = {"false": 0, "true": 1, "half": 2}#RAWFC

def get_label_list(label_ids=LABEL_IDS):
    return [k for k,v in label_ids.items()]

# def loss_func(criterion, y_true, y_pred, mask=None):
#     loss_ = criterion(y_true, y_pred)
def loss_func(criterion, y_pred, y_true, mask=None):
    loss_ = criterion(y_pred, y_true)
    if mask != None:
        # mask = tf.cast(mask, dtype=loss_.dtype)  # 将前面统计的是否零转换成1，0的矩阵
        mask = mask.type(loss_.dtype)  # 将前面统计的是否零转换成1，0的矩阵
        loss_ *= mask     # 将正常计算的loss加上mask的权重，就剔除了padding 0的影响 
    return loss_.mean()
#top_k=4
def evaluate_model(model, data_url, bsl_model=None, criterion=None, sent_criterion=None, doc_criterion=None, log=None, top_k=4, report_each_claim=12):
    """ evaluating Fact-checking model on data_url

    Args:
        model: Fact-checking model
        data_url: url to test dataset for evaluating, e.g., 'ruling_oracles_val.tsv'
        bsl_model: saved Fact-checking model
        log: logging

    Returns:
        ret: dict of accuracy, precision, recall, and f1

    """
    if log == None:
        date_str = str(datetime.now()).split('.')[0].replace(' ', '_').replace(':', '')
        LOG_FILE = pjoin(os.path.dirname(os.path.abspath(__file__)) + "/dataset/logs", f'{date_str}_exp_fc_test.log')
        log = SimpleLogger(LOG_FILE, level='info', fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if criterion == None:
        criterion = nn.CrossEntropyLoss(reduction='none')
        # sent_criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([8]).to(device))
        # pub
        # sent_criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([10]).to(device))
        # rawfc
        # sent_criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([18]).to(device))
        sent_criterion = nn.BCELoss(reduction='none', weight=torch.tensor([18]).to(device))
        doc_criterion = nn.BCELoss(reduction='none')
    log.logger.info("\nevaluating model on: %s \n", data_url)
    # dataset = End2EndDataset(data_url, next(model.parameters()).device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = myDataset(pjoin(ROOT_PROJ_PATH, data_url))
    dataset = myDataset(data_url, report_each_claim=report_each_claim)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.my_collate)#200
    ret = {'precision': 0, 'recall': 0, 'f1': 0}

    sentence_true_list, sentence_pred_list = list(), list()
    claim_true_list, claim_pred_list = list(), list()
    claim_true_count, claim_pred_count = 0, 0
    gold, pred = [], []#explain
    sent_all_pred, sent_all_true = [], []

    doc_all_pred, doc_all_true = [], [] # 选择包含oracle sents的所有报道
    gold_src_sents = []

    # for data, sentence_labels, region_labels, records_list in loader:
    eval_loss = 0.
    eval_acc = 0.
    sent_acc = 0.

    doc_acc = 0

    # switch to eval mode
    model.eval()
    with torch.no_grad():

        for oracle_ids, label_ids, raw_text_dict, lm_ids_dict in tqdm(loader):
            # claim_tensors, just_tensors, src_tensors, 

            if bsl_model:
                # pred_veraciy, (pred_evi_logits, sort_indices, selected_ids, evi_masks) = bsl_model.forward(oracle_ids, label_ids, lm_ids_dict)
                pred_veraciy, selected_indices_list, (pred_evi_logits, selected_ids, selected_sent_repr_mask, batch_sel_thresholds) = bsl_model.forward(oracle_ids, label_ids, lm_ids_dict)
                pred_veraciy = torch.softmax(pred_veraciy, dim=-1)
                # raw_text_dict['_SRC_TOK'] raw_text_dict['_TGT_TOK']
                gold_src_sents = raw_text_dict['_TGT_TOK']#[[' '.join(s) for s in sent_list] for sent_list in raw_text_dict['_TGT_TOK']]
            else:
                try:
                    # pred_veraciy, (pred_evi_logits, sort_indices, selected_ids, evi_masks) = model.forward(oracle_ids, label_ids, lm_ids_dict)
                    pred_veraciy, selected_indices_list, (pred_evi_logits, selected_ids, selected_sent_repr_mask, batch_sel_thresholds) = model.forward(oracle_ids, label_ids, lm_ids_dict)
                    # (batch_size, n_classes)
                    # sort_indices 排序后的TOP_N_DOC下标
                    pred_veraciy = torch.softmax(pred_veraciy, dim=-1)
                    # pred_veraciy_labels = torch.argmax(pred_veraciy, dim=-1)###000
                    #  (batch_size, 1)

                    # gold_src_sents = [[' '.join(raw_text_dict['_SRC_TOK'][d][id]) for id in oids[:top_k]] for d,oids in enumerate(oracle_ids)]
                    gold_src_sents = raw_text_dict['_TGT_TOK']#[[' '.join(s) for s in sent_list] for sent_list in raw_text_dict['_TGT_TOK']]
                except RuntimeError:
                    log.logger.error("all 0 tags, no evaluating at this epoch")
                    continue

            #######################
            loss = loss_func(criterion, pred_veraciy, label_ids)
            # oracle_ids = oracle_ids[:,:TOP_K]
            # doc classification TOP-5 to do 文本级
            gold_doc_labels = lm_ids_dict['gold_doc_labels']
            gold_doc_masks = lm_ids_dict['gold_doc_masks'] 
            # TASK 1 Prerparing binary ids for docs
            pred_doc_ids = [[0]*len(c_docs) for c_docs in gold_doc_labels] 
            for idx,doc_ids in enumerate(selected_indices_list):
                for idy in doc_ids:
                    pred_doc_ids[idx][idy] = 1
            # cal doc loss for each claim
            doc_loss = []
            for pred_doc,doc_label,masks in zip(pred_doc_ids, gold_doc_labels, gold_doc_masks):
                doc_loss.append(loss_func(doc_criterion, torch.FloatTensor(pred_doc).to(device), torch.FloatTensor(doc_label).to(device)))#, mask=masks
            doc_loss = sum(doc_loss)

            # TASK 2 #sel_oracle_ids (4, 5, n_sents) from oracle_ids
            gold_ids = []
            selected_docs = []
            for ids,oracle,raw_text in zip(selected_indices_list, oracle_ids, raw_text_dict['_SRC_TOK']):
                gold_ids.append(torch.vstack([oracle[id] for id in ids]))
                selected_docs.append([raw_text[i] for i in ids])

            # Sentence Classification loss
            # for i, oid in enumerate(zip(sel_oracle_ids, pred_evi_logits))
            evi_loss = []
            for pred_evi,oracle,masks in zip(pred_evi_logits, gold_ids, selected_sent_repr_mask):
                evi_loss.append(loss_func(sent_criterion, pred_evi, oracle.float(), mask=masks))

            evi_loss = sum(evi_loss)

            ## 把循环batch, 得到对应的gold ids 和 pred ids
            pred_ids = []
            for pids,oids in zip(selected_ids, gold_ids):
                _tmp = torch.zeros_like(oids, dtype=torch.float32)
                for idx,idys in enumerate(pids):
                    for idy in idys:
                        _tmp[idx][idy] = 1
                pred_ids.append(_tmp.to(device))
            
            ########先挑文本doc，再挑对应的文本里面的句子
            for  sel_docs, sel_ids, gold_doc in zip(selected_docs, selected_ids, gold_src_sents):
                gold.append(gold_doc)
                _pred = []
                for  docs, sent_ids in zip(sel_docs, sel_ids):                      
                    
                    # for j in selected_ids[i][:len(raw_text_dict['_SRC_TOK'][i])]:
                    for j in [i for i in set(sent_ids)]:
                        if j >= len(docs):
                            continue
                        # candidate = ' '.join(raw_text_dict['_SRC_TOK'][i][j])
                        _pred.append(docs[j])
                        if len(_pred) == TOP_K:
                            break
                pred.append(' \n '.join(_pred))                
                    # gold.append('\n'.join(gold_src_sents[i]))

            # loss = loss_func(criterion, pred_veraciy, label_ids)  
            # evi_loss = loss_func(sent_criterion, pred_evi_logits.reshape(-1, pred_evi_logits.shape[-1]), gold_ids.float(),  evi_masks.reshape(-1, evi_masks.shape[-1]))###
            #  (batch_size, n_tags)
            
            pred_veraciy_labels = torch.argmax(pred_veraciy, dim=1)
            # (batch_size, 1)
            eval_loss += loss.item() + evi_loss.item() + doc_loss.item()

            eval_acc += (pred_veraciy_labels == label_ids).float().mean()

            # claim_labels = region_labels.view(-1).cpu()
            claim_true_count += int((label_ids >= 0).sum())
            # # claim_pred_count += int((pred_region_labels > 0).sum())
            claim_pred_count += int((pred_veraciy_labels == label_ids).sum())

            # pred_sentence_labels = pred_sentence_labels.view(-1).cpu()
            # sentence_labels = sentence_labels.view(-1).cpu()
            # for tv, pv, in zip(sentence_labels, pred_sentence_labels):
            #     sentence_true_list.append(tv)
            #     sentence_pred_list.append(pv)

            # doc selection,  similar to exp selection
            doc_acc += mean([(pids == gids) for pids,gids in zip(pred_doc_ids, gold_doc_labels)])
            doc_all_pred.extend([torch.FloatTensor(pids).to(device) for pids in pred_doc_ids])
            doc_all_true.extend([torch.FloatTensor(gids).to(device) for gids in gold_doc_labels])

            # explain 
            sent_acc += mean([(pids == gids.cpu().tolist()) for pids,gids in zip(pred_ids, gold_ids)])
            # sent_all_pred += pred_ids.view(-1).tolist()
            # sent_all_true += gold_ids.view(-1).tolist()
            sent_all_pred.extend([pids.view(-1) for pids in pred_ids])
            sent_all_true.extend([gids.view(-1) for gids in gold_ids])
            
            # claim
            claim_true_labels = label_ids.view(-1).tolist()
            claim_pred_labels = pred_veraciy_labels.view(-1).tolist()
            # claim_true_list.extend(claim_true_labels)
            # claim_pred_list.extend(claim_pred_labels)
            for tv, pv, in zip(claim_true_labels, claim_pred_labels):
                claim_true_list.append(tv)
                claim_pred_list.append(pv)

        log.logger.info(f'Total Loss: {eval_loss / len(loader):.6f}, Veracity Acc: {eval_acc / len(loader):.6f}, Explain Acc: {sent_acc / len(loader):.6f}, Doc Acc: {doc_acc / len(loader):.6f}')
        
        # doc selection
        doc_all_true = pad_sequence(doc_all_true, batch_first=True).cpu().numpy()
        doc_all_pred = pad_sequence(doc_all_pred, batch_first=True).cpu().numpy()

        log.logger.info("Doc Classification:\n %s", classification_report(doc_all_true, doc_all_pred, labels=[0, 1], 
                                    target_names=["doc 0", "doc 1"], digits=4))


        # ROUGE: explanation generation
        # rouges = compute_rouge_from_array([' \n '.join(pred)], gold)
        rouges = compute_rouge_from_array(pred, gold)
        log.logger.info('Rouges = \n%s' % (rouge_results_to_str(rouges)))
        
                # tensor to numpy or list
        sent_all_true = pad_sequence(sent_all_true, batch_first=True).cpu().numpy()
        sent_all_pred = pad_sequence(sent_all_pred, batch_first=True).cpu().numpy()
        log.logger.info("explanation generation result:\n %s", classification_report(sent_all_true, sent_all_pred, labels=[0, 1], 
                                    target_names=["class 0", "class 1"], digits=6))

        f1_sent = f1_score(sent_all_true, sent_all_pred, average='macro')
        precision_sent = precision_score(sent_all_true, sent_all_pred, average="macro")
        recall_sent = recall_score(sent_all_true, sent_all_pred, average="macro")

        log.logger.info(f'P_sent: {precision_sent}, R_sent: {recall_sent}, F1_sent: {f1_sent}')

        # veracity
        macrof = classification_report(claim_true_list, claim_pred_list,
                              target_names=get_label_list(LABEL_IDS),
                              digits=6).split('\n')[-3].split()[-2]

        log.logger.info("claim classification result:\n %s", classification_report(claim_true_list, claim_pred_list,
                                    target_names=get_label_list(LABEL_IDS), digits=6))

        ret = dict()
        tp = 0
        for pv, tv in zip(claim_pred_list, claim_true_list):
            # if pv == tv == 0:
            #     continue
            if pv == tv:
                tp += 1
        # fp = claim_pred_count - tp
        # fn = claim_true_count - tp
        fp = len(claim_pred_list) - tp
        fn = claim_true_count - tp

        ret['precision'], ret['recall'], ret['f1'] = calc_f1(tp, fp, fn)
        ret['macrof'] = eval(macrof)

        ret['rouges'] = rouges
        ret['precision_sent'] = precision_sent
        ret['recall_sent'] = recall_sent
        ret['f1_sent'] = f1_sent

    return ret


def main():
    
    model_url = from_project_root("./LIAR-RAW_model_xxxx.pt")#
    # test_url = from_project_root("dataset/pub_oracles/ruling_oracles_test_checked.tsv")
    test_url = from_project_root("dataset/LIAR-RAW/test.json")
    model = torch.load(model_url)
    # from sentence_transformers import SentenceTransformer, util
    # tese_precision, test_recall, test_micf1, test_macrof, test_rouges, test_p_sent, test_r_sent, test_f1_sent = evaluate_model(model, test_url, lm_emb = SentenceTransformer('paraphrase-MiniLM-L6-v2')).values()
    tese_precision, test_recall, test_micf1, test_macrof, test_rouges, test_p_sent, test_r_sent, test_f1_sent = evaluate_model(model, test_url).values()
    print("results on the test set: P_sent %6f, R_sent %6f, macF_sent %6f.\n %s \n  Finished!" % (test_p_sent, test_r_sent, test_f1_sent, rouge_results_to_str(test_rouges)))

if __name__ == '__main__':
    main()
