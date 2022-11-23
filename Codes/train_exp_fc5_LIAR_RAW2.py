# encoding: utf-8
# fine-tune  DistilEmbeddings
# coling221030 @zwyang
from transformers import set_seed
from transformers import get_scheduler
from statistics import mean
import os
from os.path import exists
import time
from datetime import datetime
import json
import torch
# import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from os.path import join as pjoin

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from helpers.reader5 import load_df_dataset
from helpers.reader5 import myDataset
from helpers.utils import rouge_results_to_str, compute_rouge_from_array

import sys
sys.path.append('helpers/')
sys.path.append('model/')
# from logger import Logger
# from model import BasicFC
from model_exp_fc5 import ExplainFC #DistilEmbeddings

# from dataset.oracles.vocab_process import preprocess
from helpers.path_util import from_project_root, dirname

# from eval import evaluate_model
from eval_exp_fc5 import evaluate_model, loss_func

from helpers.simple_logger import SimpleLogger
os.environ["CUDA_VISIBLE_DEVICES"] = '1'#3
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

torch.manual_seed(100)
set_seed(100)

REPORT_EACH_CLAIM = 30 #Totally retrive how many reprots for each claim
sent_dim = 768#384#768
num_prerun = 1

batch_size = 2
learning_rate = 1e-5 #0.0005#1e-3
n_epochs = 8#30
EMBED_URL = "dataset/oracles/embeddings.npy"

ROOT_PROJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/CofCED/dataset/LIAR-RAW"
output_vocab = "vocab.json"
output_char_vocab = "char_vocab.json"

output_vocab_article_source = 'vocab_article_source.json'
FREEZE_WV = False
EARLY_STOP = 3#5
MAX_GRAD_NORM = 5
N_TAGS = 6
date_str = str(datetime.now()).split('.')[0].replace(' ', '_').replace(':', '')
LOG_FILE = pjoin(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/CofCED/dataset/logs", f"{date_str}_DOC_ExplainFC5-DistilBERT_auto_{ROOT_PROJ_PATH.split('/')[-1]}2-all.log")

def to_np(x):
    if x == None:
        return x.cpu().data.numpy()
    else:
        return x

def get_dynamicWA(claim_loss, sent_loss, report_loss, step=1, claim_labels=6, sent_labels=2, temperature=8.0, K=1):
    '''dynamic weight average for multi-task learning loss
    temperature: larger value results in a more even distribution between different tasks. 
    '''
    if step <= 2:
        return 1,1,1
    C1, C2 = claim_loss[-1], claim_loss[-2]

    S1, S2 = max(sent_loss[-1], torch.full_like(sent_loss[-1], 0.01)), max(sent_loss[-2], torch.full_like(sent_loss[-1], 0.01))
    D1, D2 = max(report_loss[-1], torch.full_like(report_loss[-1], 0.01)), max(report_loss[-2], torch.full_like(report_loss[-1], 0.01))
    # 

    if torch.isnan(S1) or torch.isnan(S2):
        return 1,1,1
    if torch.isnan(D1) or torch.isnan(D2):
        return 1,1,1

    if S1 < 0 or S2 < 0:
        return 1,1,1
    if D1 < 0 or D2 < 0:
        return 1,1,1

    # add time param
    c_t1 = get_time_param(step-3)* (C1 / C2)/temperature
    s_t1 = get_time_param(step-3)* (S1 / S2)/temperature
    d_t1 = get_time_param(step-3)* (D1 / D2)/temperature

    # cret sret
    csd_ret = F.softmax(torch.FloatTensor([c_t1, s_t1, d_t1]), dim=-1) * K 
    param1= csd_ret.data.numpy()[0]
    param2= csd_ret.data.numpy()[1]
    param3= csd_ret.data.numpy()[2]
    return param1, param2, param3

def get_time_param(x):
    '''convex function:fast->slow, characterizing the easier classification for early phase, and harder classification for later phase. BIG HEAD'''
    y = math.log(x+1,2) 
    return y

def train_model(n_epochs=n_epochs,
                train_file = "train.json",##containing all data
                val_file = "val.json",
                test_file = 'test.json',
                embedding_url=EMBED_URL,
                char_feat_dim=0,#20
                source_dim = 20,#20
                freeze=FREEZE_WV,
                learning_rate=learning_rate,
                batch_size=batch_size,
                early_stop=EARLY_STOP,
                clip_norm=MAX_GRAD_NORM,
                n_tags=N_TAGS,
                saved_model_url=None,
                save_only_best=True,
                log=LOG_FILE,
                vocab_article_source=output_vocab_article_source,
                REPORT_EACH_CLAIM=REPORT_EACH_CLAIM,
                TOP_K=4):


    # print arguments
    arguments = json.dumps(vars(), indent=2)
    log = SimpleLogger(LOG_FILE, level='info', fmt='%(asctime)s - %(levelname)s: %(message)s')
    log.logger.info(LOG_FILE)
    log.logger.info("arguments %s", arguments)
    start_time = datetime.now()

    # load_df_dataset
    # filename = pjoin(ROOT_PROJ_PATH, f'ruling_oracles_{train_file}.tsv')
    filename = pjoin(ROOT_PROJ_PATH, train_file)
    # lm_emb = DistilEmbeddings()
    # dataset = myDataset(filename, lm_emb, report_each_claim=REPORT_EACH_CLAIM)
    dataset = myDataset(filename, report_each_claim=REPORT_EACH_CLAIM)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.my_collate)

    model = ExplainFC(
        hidden_size=384,#200
        lstm_layers=1,
        n_tags=N_TAGS,
        char_feat_dim=char_feat_dim,
        embedding_url=EMBED_URL,
        bidirectional=True,
        n_embeddings=1000,
        embedding_dim=300, #768,#300
        lm_embedding_dim = sent_dim,
        freeze = FREEZE_WV,
        max_doc_num=REPORT_EACH_CLAIM,#30，12
        vocab_article_source=None,#pjoin(ROOTPATH,vocab_article_source),
        source_dim = source_dim
    )
    # continue
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.logger.info("Welcome using %s", device)

    model = model.to(device)
    saved_model = torch.load(saved_model_url) if saved_model_url else None

    # loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')
    sent_criterion = nn.BCELoss(reduction='none', weight=torch.tensor([8]).to(device))

    doc_criterion = nn.BCELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=50, num_training_steps=len(train_loader)*n_epochs)


    cnt = 0
    max_f1, max_f1_epoch = 0, 0
    max_rouge, max_rouge_epoch = 0, 0
    best_model_url = None
    claim_loss = []
    sent_loss = []
    report_loss = []
    # begin training
    for epoch in range(n_epochs):
        log.logger.info('*' * 10)
        log.logger.info(f'epoch {epoch + 1}')
        since = time.time()
        running_loss = 0.0
        running_acc = 0.0
        evi_running_acc = 0.0

        train_acc = 0.0
        train_precision = 0.0
        train_recall = 0.0
        train_f1 = 0.0
        model.train()

        for i, (oracle_ids, label_ids, raw_text_dict, lm_ids_dict) in enumerate(train_loader, 1):

            optimizer.zero_grad()
            # pred_veraciy, (pred_evi_logits, sort_indices, selected_ids, evi_masks) = model.forward(oracle_ids, label_ids, lm_ids_dict)
            pred_veraciy, selected_indices_list, (pred_evi_logits, selected_ids, selected_sent_repr_mask, batch_sel_thresholds) = model.forward(oracle_ids, label_ids, lm_ids_dict)
            # cla_loss = criterion(pred_veraciy, label_ids)
            cla_loss = loss_func(criterion, pred_veraciy, label_ids)
            # oracle_ids = oracle_ids[:,:TOP_K]

            # doc classification TOP-5 to do DOC DOC
            gold_doc_labels = lm_ids_dict['gold_doc_labels']
            gold_doc_masks = lm_ids_dict['gold_doc_masks'] 
            # TASK 1 Preparing binary ids for docs
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
            for ids,oracle in zip(selected_indices_list, oracle_ids):
                gold_ids.append(torch.vstack([oracle[id] for id in ids]))

            # Sentence classification loss
            evi_loss = []
            for pred_evi,oracle,masks in zip(pred_evi_logits, gold_ids, selected_sent_repr_mask):
                evi_loss.append(loss_func(sent_criterion, pred_evi, oracle.float(), mask=masks))
            evi_loss = sum(evi_loss)

            ## batch, obtaining gold_ids and pred_ids
            pred_ids = []
            for pids,oids in zip(selected_ids, gold_ids):
                _tmp = torch.zeros_like(oids, dtype=torch.float32)
                for idx,idys in enumerate(pids):
                    for idy in idys:
                        _tmp[idx][idy] = 1
                pred_ids.append(_tmp.to(device))


            #  Dynamic Weight Average for MT loss
            param1, param2, param3 = get_dynamicWA(claim_loss, sent_loss, report_loss, i, claim_labels=N_TAGS, temperature=8, K=2)
            loss = param1*cla_loss + param2*evi_loss + param3*doc_loss

            # append loss for update params
            claim_loss.append(cla_loss.detach())
            sent_loss.append(evi_loss.detach())
            report_loss.append(doc_loss.detach())

            running_loss += loss.item()
            _, pred = torch.max(pred_veraciy, 1)####
            
            accuracy = (pred == label_ids).float().mean()
            running_acc += accuracy.item()
            # train_acc += accuracy_score(label_ids.cpu(), pred.cpu())
            train_precision = precision_score(label_ids.cpu(), pred.cpu(), average="macro")
            train_recall = recall_score(label_ids.cpu(), pred.cpu(), average="macro")
            train_f1 = f1_score(label_ids.cpu(), pred.cpu(), average="macro")
                        
            # explanation
            # evi_accuracy = (pred_ids == gold_ids).float().mean() # evi_accuracy = mean([(pids == gids.cpu().tolist()) for pids,gids in zip(pred_ids, gold_ids)])
            evi_accuracy = mean([torch.mean((pids == gids).float()).item() for pids,gids in zip(pred_ids, gold_ids)])
            evi_running_acc += evi_accuracy #.item()
            #pred_evi_logits.round()
            # evi_train_precision = precision_score(gold_ids.reshape(-1).cpu(), pred_ids.reshape(-1).cpu().detach(), average="macro")
            evi_train_precision = mean( [precision_score(gids.reshape(-1).cpu(), pids.reshape(-1).cpu().detach(), average="macro") for pids,gids in zip(pred_ids, gold_ids)] )
            evi_train_recall = mean( [recall_score(gids.reshape(-1).cpu(), pids.reshape(-1).cpu().detach(), average="macro") for pids,gids in zip(pred_ids, gold_ids)] )
            evi_train_f1 = mean( [f1_score(gids.reshape(-1).cpu(), pids.reshape(-1).cpu().detach(), average="macro") for pids,gids in zip(pred_ids, gold_ids)] )


            loss.backward()
            # gradient clipping
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()
            # if i % 4 == 0:
            #     optimizer.step()
            lr_scheduler.step()

            if i % 20 == 0:
            # log.logger.info(f'[{epoch + 1}/{n_epochs}] Loss: {running_loss / (batch_size * i):.6f}, Acc: {running_acc / (batch_size * i):.6f}')
                # log.logger.info(f'Ep[{epoch + 1}]-Batch[{i}/{len(train_loader)}] Loss: {loss:.6f}, Ac: {accuracy:.6f}, p: {train_precision:.6f}, r: {train_recall:.6f}, maf1: {train_f1:.6f}|| eA: {evi_accuracy:.6f}, eP: {evi_train_precision:.6f}, eR: {evi_train_recall:.6f}, emaf1: {evi_train_f1:.6f}, [{round(param1, 2)}:{round(param2, 2)}]')
                log.logger.info(f'Ep[{epoch + 1}]-Batch[{i}/{len(train_loader)}] Loss: {loss:.4f} [{cla_loss:.2f}:{evi_loss:.2f}:{doc_loss:.2f}], Ac: {accuracy:.4f}, p: {train_precision:.4f}, r: {train_recall:.4f}, maf1: {train_f1:.6f}|| eA: {evi_accuracy:.4f}, eP: {evi_train_precision:.4f}, eR: {evi_train_recall:.4f}, emaf1: {evi_train_f1:.6f}')

            # ========================= Log ======================
            # if i % 30 == 0:
            #     print(f'[{epoch + 1}/{n_epochs}] Loss: {running_loss / (batch_size * i):.6f}, Acc: {running_acc / (batch_size * i):.6f}')
            #     print(f'Acc:{train_acc / i:.6f}, prec:{train_precision / i:.6f}, recall:{train_recall / i:.6f}, f1:{train_f1 / i:.6f}')

        log.logger.info(f'Finish {epoch + 1} epoch, Loss: {running_loss / i:.6f}, Acc: {evi_running_acc / i:.6f}')
        # print(f'Acc:{train_acc / i:.6f}, prec:{train_precision / i:.6f}, recall:{train_recall / i:.6f}, f1:{train_f1 / i:.6f}')

        # val_file: evaling model.eval()
        cnt += 1
        precision, recall, f1, macrof, rouges, p_sent, r_sent, f1_sent = evaluate_model(model, pjoin(ROOT_PROJ_PATH, val_file), saved_model, log=log, report_each_claim=REPORT_EACH_CLAIM).values()
        print('OOOOh my god, buy it! zzwzw', p_sent, r_sent, f1_sent)
        if macrof > max_f1:
            # max_rouge, max_rouge_epoch = rouges["rouge2_f1"], epoch+1 if rouges["rouge2_f1"] > max_rouge else max_rouge, max_rouge_epoch
            max_f1, max_f1_epoch = macrof, epoch+1
            name = 'split' if saved_model else ROOT_PROJ_PATH.split('/')[-1] #'ExplainFC'
            if save_only_best and best_model_url:
                os.remove(best_model_url)
            # best_model_url = from_project_root("%s_model_epoch%d_%f.pt" % (name, epoch, f1))
            best_model_url = from_project_root("%s_model_epoch%d_pre%f_rec%f_micf%f_macf%f_rouge%f.pt" % (name, epoch, precision, recall, f1, macrof, rouges["rouge2_f1"]))
            torch.save(model, best_model_url)
            cnt = 0

        log.logger.info("maximum of f1 value: %.6f, in epoch #%d" % (max_f1, max_f1_epoch))
        log.logger.info("training time: %s", str(datetime.now() - start_time).split('.')[0])
        log.logger.info(datetime.now().strftime("%c\n"))

        if cnt >= early_stop > 0:
            break

    if test_file:
        best_model = torch.load(best_model_url)
        log.logger.info("best model url: %s", best_model_url)
        log.logger.info("evaluating on test dataset: %s", test_file)
        tese_precision, test_recall, test_micf1, test_macrof, test_rouges, test_p_sent, test_r_sent, test_f1_sent = evaluate_model(best_model, pjoin(ROOT_PROJ_PATH, test_file), saved_model, log=log, report_each_claim=REPORT_EACH_CLAIM).values()
        log.logger.info("results on the test set: P %6f, R %6f, micF %6f, macF %6f.\n P_sent %6f, R_sent %6f, macF_sent %6f.\n  Finished!" % (tese_precision, test_recall, test_micf1, test_macrof, test_p_sent, test_r_sent, test_f1_sent))
    log.logger.info(arguments)
    log.logger.info(LOG_FILE)

    # save
    # date_str = str(datetime.now()).split('.')[0].replace(' ', '_').replace(':', '')
    # torch.save(model.state_dict(), f'./{20210613}_FactChecking.pt')

if __name__ == '__main__':
    # start_time = datetime.now()
    if EMBED_URL and not exists(EMBED_URL):
        # pre-trained embedding url, word2vec format
        print("preprocess vocabulary...")
        # preprocess(readfile, output_vocab, output_char_vocab, PRETRAINED_URL)
    train_model()
    print("finished in:", datetime.now())
