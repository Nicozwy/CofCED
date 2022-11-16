## read snopes, with multiple docs
import json
from nltk.tokenize import word_tokenize
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import pandas as pd
import os
from os.path import join as pjoin
from helpers.path_util import from_project_root, dirname
import helpers.json_util as ju

from tqdm import tqdm
import pickle

# from helpers.lm_embeddings import data2ids, list2str
import sys
# sys.path.append('gen_emb/')
# from gen_emb.albert_emb import PreEmbeddedLM
# from gen_emb.distilbert_emb import list2str

from transformers import DistilBertTokenizer
import nltk
#LIAR-PLUS
ROOT_PROJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/dataset/LIAR-RAW"
# DATA_FILE_PATH = "..\\dataset\\oracles"#six-class
# PRETRAINED_URL = from_project_root("dataset/embedding/glove.840B.300d.word2vec.vocab")
VOCAB_EMB_URL = 'embeddings.npy'##generated from PRETRAINED
VOCAB_URL = "vocab.json"
CHAR_VOCAB_URL = "char_vocab.json"

# MAX_ORACLE= 55 # 最多的oracle的句子数

# LABEL_IDS = {"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5}
# LABEL_LIST = {"pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"}
# LABEL_LIST = {"false", "half", "true"}
global_max_claimnum = 2
# global_max_srcnum = 100
if 'pub' in ROOT_PROJ_PATH:
    LABEL_IDS = {"false": 0, "true": 1, "mixture": 2, "unproven": 3}
elif 'fever' in ROOT_PROJ_PATH:
    # LABEL_IDS = {"REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2}
    LABEL_IDS = {"REFUTES": 0, "SUPPORTS": 1}
elif 'RAWFC' in ROOT_PROJ_PATH or 'seefact' in ROOT_PROJ_PATH:
    LABEL_IDS = {"false": 0, "true": 1, "half": 2}
elif 'LIAR-RAW' in ROOT_PROJ_PATH:
    LABEL_IDS = {"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5}
else:
    LABEL_IDS = {"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5}

TOP_K_ORACLE_NUMS = 5#prepare 5 FOR PUB_ORACLE , 5 FOR LIAR-PLUS

class myDataset(Dataset):
    # def __init__(self, data_file, lm_emb, report_each_claim=30):
    def __init__(self, data_file, report_each_claim=30):
        super().__init__()
        # self.df = pd.read_csv(csv_file, sep='\t')
        # json obj list
        self.df = self.read_from_dir(data_file)
        
        self.finetune = False#False
        self.report_each_claim = report_each_claim

        self._len = len(self.df)        
        # self.lm_emb = PreEmbeddedLM(f'./dataset/oracles/albert.emb.pkl')
        # self.lm_emb = DistilEmbeddings() #SentenceTransformer('paraphrase-MiniLM-L6-v2')

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')#("bert-base-uncased")
        # self.embedding_cache_path = from_project_root(ROOT_PROJ_PATH+"/pub.emb.pkl")##.format(dataset)
        # self.dataset = self.generate_beans(self.df)
        # self._x, self._justs, self._y = self.generate_beans(self.df)
        # self._claim, self._claim_tok, self._just, self._just_tok, self._src, self._src_tok, self._oracle_ids, self._y = self.load_raw(self.df)
        self.event_id, self.claim, self.label, \
            self.explain, self.report_links, self.report_contents, \
                self.report_domains, self.tok_sents, self.tok_sent_ids = self.load_raw(self.df)

    def __getitem__(self, index):
        # train_loader 接受数据
        # return self._claim[index], self._claim_tok[index], self._just[index], self._just_tok[index], self._src[index], self._src_tok[index], self._oracle_ids[index], self._y[index]
        return self.event_id[index], self.claim[index], self.label[index], \
            self.explain[index], self.report_links[index], self.report_contents[index], \
                self.report_domains[index], self.tok_sents[index], self.tok_sent_ids[index]

    def __len__(self):
        return self._len

    def read_from_dir(self, path):
        ''''1 csv pd.read_csv
            2 json return all filenames in dir'''

        # if os.path.isfile(path) and '.csv' in path:
        #     return pd.read_csv(path, sep='\t')
        if os.path.isfile(path) and 'LIAR-RAW' in path:
            with open(path, 'r', encoding='utf-8') as f:
                all_data = json.load(f) # all_data is a list contain all data.
            return all_data
        else: 
            # 得到文件夹下所有json文件的名称
            filenames = os.listdir(path)
            name_list = []
            for name in filenames:
                if '.json' in name: 
                    name_list.append(name)

            # 读取所有json
            if len(name_list) == 1:
                all_data = ''
                # for fever, only allow 1 file in {mode} dir 
                for file in name_list:
                    filename = pjoin(path, file)# root/xxxx.json
                    with open(filename, 'r', encoding='utf-8') as json_file:
                        all_data = json.load(json_file)
            else:
                all_data = []
                # for our snope, allow many files in {mode} dir
                for file in name_list:
                    filename = pjoin(path, file)# root/xxxx.json
                    with open(filename, 'r', encoding='utf-8') as json_file:
                        obj = json.load(json_file)
                        all_data.append(obj)
            return all_data

    def load_raw(self, df):
        '''parsing dict objs to list '''
        raw_data = [[] for _ in range(9)]##event_id, claim, label, explain, (link, content, domain, report_sents, report_is_evidence) 0/1      
        for obj in tqdm(df):
            report_tok_sents = []           
            report_tok_sent_ids = []  
       
            report_links = []
            report_contents = []
            report_domains = []  
            # keys: event_id, claim, original_label, label, explain, reports
            # event_id, claim, _, label, explain, reports = obj.values()
            event_id, claim, label, explain, reports = obj['event_id'], obj['claim'], obj['label'], obj['explain'], obj['reports']

            raw_data[0].append(event_id)
            raw_data[1].append(claim)
            # raw_data[i].append(original_label)
            raw_data[2].append(label)
            raw_data[3].append(explain)
            # raw_data[4].append(reports) 
            for s in reports[:self.report_each_claim]: # 截取
                report_links.append(s['link']) 
                report_contents.append(s['content']) #全文, doc
                report_domains.append(s['domain']) 
                # for tokenized sents
                tok_sents = [] 
                tok_sent_ids = []   
                for ts in s['tokenized']:
                    tok_sents.append(ts['sent'])
                    tok_sent_ids.append(ts['is_evidence'])
                    
                report_tok_sents.append(tok_sents)
                report_tok_sent_ids.append(tok_sent_ids)
            raw_data[4].append(report_links) 
            raw_data[5].append(report_contents) #全文, doc sent_list
            raw_data[6].append(report_domains) 

            raw_data[7].append(report_tok_sents) 
            raw_data[8].append(report_tok_sent_ids) 

        return raw_data

    def my_collate(self, batch):
        '''collect data with your style'''
        # event_id, claim, label, explain, (link, content, domain, report_sents, report_is_evidence)
        event_id, claim, label, \
            explain, link, content, \
                domain, report_sents, report_is_evidence = [[] for _ in range(9)]
        raw_data_list = []##event_id, claim, label, explain, (link, content, domain, report_sents, report_is_evidence) 0/1 

        raw_text_dict = {}
        lm_ids_dict = {}

        lm_embs_dict = {}

        num_report_docs = []
        for i,item in enumerate(batch):
            # raw_data_list[i].append(item[i])
            event_id.append(item[0])
            claim.append(item[1])
            # label.append(LABEL_IDS[item[2]])
            label.append(item[2])

            explain.append(item[3])
            link.append(item[4])
            content.append(item[5])
            # save number of reports
            num_report_docs.append(len(item[5]))

            domain.append(item[6])
            report_sents.append(item[7])
            report_is_evidence.append(item[8])

        raw_data_list.append(event_id)
        raw_data_list.append(claim)
        raw_data_list.append(label)
        raw_data_list.append(explain)
        raw_data_list.append(link)
        raw_data_list.append(content)
        raw_data_list.append(domain)
        raw_data_list.append(report_sents)
        raw_data_list.append(report_is_evidence)

        lm_ids_dict['num_report_docs'] = num_report_docs

        # claim, just, ruling comments
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # claim_tensors, just_tensors, src_tensors, _CLAIM_TOK, _SRC_TOK, src_sent_num, report_domains = gen_sent_tensors(raw_data_list, device=_device, npy_path=VOCAB_EMB_URL)# gen sentence tensors for claims
        # claim label
        label_ids = gen_label_id(raw_data_list[2], device=_device)
        # needing mask
        oracle_ids = gen_oracle_id(raw_data_list[-1], device=_device)

        # num of each report
        num_oracle_list = [torch.sum(nums, dim=1) for nums in oracle_ids]
        num_oracle_list.insert(0, torch.zeros(self.report_each_claim))
        num_oracle_eachdoc = pad_sequence(num_oracle_list, batch_first=True)
        num_oracle_eachdoc = num_oracle_eachdoc[1:]
        # (batch_size, )

        # Task: doc label : whehter to select this report for evidence
        lm_ids_dict['label_ids'] = label_ids

        lm_ids_dict['gold_doc_labels'] = [[int(bool(sum(doc))) for doc in claim_docs] for claim_docs in raw_data_list[8]]
        lm_ids_dict['gold_doc_masks'] = torch.FloatTensor([[True]*t + [False]*(num_oracle_eachdoc.shape[1] - t) for t in num_report_docs]).to(_device)

        # return raw_text for ROUGE computate
        # event_id, claim, label,  explain, link, content,  domain, report_sents, report_is_evidence = raw_data_list
        raw_text_dict['_CLAIM_TOK'] = raw_data_list[1]
        # raw_text_dict['_TGT'] = _TGT
        raw_text_dict['_TGT_TOK'] = raw_data_list[3]
        # raw_text_dict['_SRC'] = _SRC
        raw_text_dict['_SRC_TOK'] = raw_data_list[7]
        # raw_text_dict['all'] = raw_data_list ### 

        lm_ids_dict['claim_ids'] = []
        lm_ids_dict['claim_masks'] = []
        lm_ids_dict['src_ids'] = []
        lm_ids_dict['src_masks'] = []
        
        lm_embs_dict['claim_embs'] = []
        lm_embs_dict['src_embs'] = []

        # 每个report 包含多少个句子
        # lm_ids_dict['src_sent_num'] = src_sent_num
        lm_ids_dict['src_sent_num'] = [[len(reports) for reports in claims] for claims in raw_data_list[7]]        
        lm_ids_dict['num_oracle_eachdoc'] = num_oracle_eachdoc
        lm_ids_dict['report_domains'] = gen_domain_id(raw_data_list[6], device=_device)
        # lm_ids_dict['report_domains_ids'] = []

        max_length = cal_max_word_num(raw_data_list)

        # max_length = max_length if max_length < 512 else 512
        max_length = max_length if max_length < 300 else 300

        # max_length2 = claim_tensors[0].shape[-1]
        # assert claim_tensors[0].shape[-1] <= 512 ## BERT max_length = 512
        # assert src_tensors[0].shape[-1] <= 512 
        for i in range(len(raw_data_list[1])):
            # claim_ids, claim_att_mask = self.lm_emb(mergelist(_CLAIM_TOK[i]), max_length=max_length)
            # src_ids, src_att_mask = self.lm_emb(_SRC_TOK[i], max_length=max_length)
            
            # with torch.no_grad(): move to distilbert_emb.py
            # claim_ids, _ = self.lm_emb.embed([' '.join(list2str(_CLAIM_TOK[i], max_length))] if len(list2str(_CLAIM_TOK[i], max_length)) > 1 else list2str(_CLAIM_TOK[i], max_length), max_length=max_length, finetune=self.finetune)                
            # claim_ids, _ = self.lm_emb.embed([raw_data_list[1][i]], max_length=max_length, finetune=self.finetune)   
            # 直接在reader.py中获取 input_ids, attention_mask
            # claim_ids, claim_attention_mask = self.data2ids([raw_data_list[1][i]], add_special_tokens=True)#max_length 这里的用法是取多少个词 （1, 12）
            # 默认每个claim 只含一个句子
            encoded_claim = self.tokenizer(raw_data_list[1][i], return_tensors='pt', padding=True, truncation=True, max_length=max_length) #, truncation=True, max_length=max_length
            claim_ids, claim_attention_mask = encoded_claim['input_ids'], encoded_claim['attention_mask']

            lm_ids_dict['claim_ids'].append(claim_ids)#repr
            lm_ids_dict['claim_masks'].append(claim_attention_mask)#repr
            # lm_ids_dict['tgt_ids'].append(tgt_ids.to(_device))

        # for i in range(len(raw_data_list[7])):与shagn'su
            # src_ids, _ = self.lm_emb.embed(list2str(_SRC_TOK[i], max_length), max_length=max_length, finetune=self.finetune)#xxxx
            # src_ids, src_attention_mask = self.data2ids(raw_data_list[7][i], max_length=max_length, add_special_tokens=True) #, max_length=max_length
            
            encoded_src_docs = [self.tokenizer(doc, return_tensors='pt', padding=True, truncation=True, max_length=max_length) for doc in raw_data_list[7][i]]#, truncation=True, max_length=max_length
            src_ids, src_attention_mask = [], []
            for doc_dict in encoded_src_docs:
                src_ids.append(doc_dict['input_ids'])
                src_attention_mask.append(doc_dict['attention_mask'])
            
            lm_ids_dict['src_ids'].append(src_ids)
            lm_ids_dict['src_masks'].append(src_attention_mask)


        # return claim_tensors, just_tensors, src_tensors, oracle_ids, label_ids, raw_text_dict, lm_ids_dict
        return oracle_ids, label_ids, raw_text_dict, lm_ids_dict


    def data2ids(self, sentences, labels=None, max_length=None, add_special_tokens=False):
        input_ids,attention_mask=[],[]
        #input_ids是每个词对应的索引idx ;token_type_ids是对应的0和1，标识是第几个句子；attention_mask是对句子长度做pad
        #input_ids=[22,21,...499] token_type_ids=[0,0,0,0,1,1,1,1] ;attention_mask=[1,1,1,1,1,0,0,0]补零
        for i in range(len(sentences)):
            encoded_dict = self.tokenizer.encode_plus(
            sentences[i],
            add_special_tokens = add_special_tokens,      #False 添加 '[CLS]' 和 '[SEP]'
            max_length = max_length,           # 填充 & 截断长度
            pad_to_max_length = True,
            truncation=True,
            return_tensors = 'pt',         # 返回 pytorch tensors 格式的数据
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_mask.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)#把多个tensor合并到一起
        attention_mask = torch.cat(attention_mask, dim=0)

        input_ids = torch.LongTensor(input_ids)#每个词对应的索引
        attention_mask = torch.LongTensor(attention_mask)#[11100]padding之后的句子
        # if labels!=None:
        #     labels = torch.LongTensor(labels)#所有实例的label对应的索引idx
            
        return input_ids, attention_mask

def mergelist(sentlist):
    if type(sentlist[0]) == list:
        # to avoid exceed the max_length of BERT---512
        ret = []
        for item in sentlist:
            ret.extend(item)
            # ret.append('[SEP]')
                
        # sentlist = [item[:512] for item in sentlist]
        return [ret]

def len_allsents(_CLAIM_TOK):
    '''@_CLAIM_TOK：list of list'''
    lens = []
    for sents in _CLAIM_TOK:
        _len = 0
        for s in sents:
            _len += len(s)#maybe more than one sents
        lens.append(_len)
    assert len(lens) == len(_CLAIM_TOK)
    return lens

def gen_oracle_id(_ORACLE_IDS, device):
    '''select oracle ids from ruling_tokenized'''
    # further work: padding 0
    # oracle_ids = torch.LongTensor(_ORACLE_IDS).to(device)
    # return oracle_ids ,i.e., is_evidence
    # _ids = [torch.LongTensor(id).to(device) for id in _ORACLE_IDS]
    # multi-doc
    doc_ids = [[torch.LongTensor(id).to(device) for id in doc_ids] for doc_ids in _ORACLE_IDS]
    oracle_ids = [pad_sequence(_ids, batch_first=True) for _ids in doc_ids]
    return oracle_ids #[:, :TOP_K_ORACLE_NUMS]

def gen_label_id(_LABEL, device, label_ids=LABEL_IDS):
    '''generate label ids for claims'''
    _ids = [label_ids[la] for la in _LABEL]
    id_tensors = torch.LongTensor(_ids).to(device)
    return id_tensors

def get_idx_list(len_list):
    ''' generate real index of the tensor.
    :param len_list: lengths of the tensor.
    :return: list of real index.
    '''
    rets = []
    _tmp = 0
    for value in len_list:
        _tmp = _tmp + value
        rets.append(_tmp)
    return rets

def gen_domain_id(domain, data_url=ROOT_PROJ_PATH, filename='vocab_article_source.json', device='cuda'):
    '''obtain domain ids'''
    source_url = pjoin(data_url, filename)### domain
    source_vocab = ju.load(source_url)
    report_domains = []
    unk_idx = 1
    for ds in domain:
        # domain item to item id
        ds_item = torch.LongTensor([source_vocab[item] if item in source_vocab else unk_idx
                                        for item in ds]).to(device)
        report_domains.append(ds_item)
    # report_domains = pad_sequence(report_domains, batch_first=True)
    # (batch_size, max_num_reports)

    return report_domains

def cal_max_word_num(raw_data):
    ''' return max len'''
    
    event_id, claim, label,  explain, link, content,  domain, report_sents, report_is_evidence = raw_data
    _CLAIM_TOK = []
    _SRC_TOK = []
    # _SOURCE_LIST = []#domain
    # for each claim
    for cl, rs in zip(claim, report_sents):
        _CLAIM_TOK.append(nltk.word_tokenize(cl))
        # multi-docs containing sents, respectively
        _SRC_TOK.append([[nltk.word_tokenize(sent) for sent in r] for r in rs])
        # _SOURCE_LIST.append(ds)

    cla_len = [len(cla) for cla in _CLAIM_TOK]# length for each sent

    src_len = []# length for each sent
    src_tok_list = []
    for tok in _SRC_TOK:
        for t in tok:
            src_tok_list.extend(t)
            src_len.extend([len(s) for s in t])
    
    return max(cla_len+src_len)

    

def gen_sent_tensors(raw_data, device='auto', data_url=ROOT_PROJ_PATH, npy_path=VOCAB_EMB_URL):
    '''generate input tensors for 1 batch： _CLAIM, _CLAIM_TOK, _TGT, _TGT_TOK, _SRC, _SRC_TOK
    Args:
        _CLAIM, _CLAIM_TOK: claim sents and its token format
        _TGT, _TGT_TOK: just and just_tokenized
        _SRC, _SRC_TOK: ruling and ruling_tokenized
        data_url=ROOT_PROJ_PATH:  parent path, e.g., pjoin(data_url, 'vocab.json')
        vocab_path: path to vocab.json
        char_vocab_path: path to char_vocab.json
        npy_path: path to embeddings.npy

    Returns:

    '''
    event_id, claim, label,  explain, link, content,  domain, report_sents, report_is_evidence = raw_data
    _CLAIM_TOK = []
    _SRC_TOK = []
    # _SOURCE_LIST = []#domain
    # for each claim
    for cl, rs in zip(claim, report_sents):
        _CLAIM_TOK.append(nltk.word_tokenize(cl))
        # multi-docs containing sents, respectively
        _SRC_TOK.append([[nltk.word_tokenize(sent) for sent in r] for r in rs])
        # _SOURCE_LIST.append(ds)

    npy_path_url = pjoin(data_url, npy_path)
    vocab_url = pjoin(data_url, VOCAB_URL)
    char_vocab_url = pjoin(data_url, CHAR_VOCAB_URL)
    # max_len = max(len_allsents(_CLAIM_TOK))
    source_url = pjoin(data_url, 'vocab_article_source.json')### domain

    vocab = ju.load(vocab_url)
    char_vocab = ju.load(char_vocab_url)
    source_vocab = ju.load(source_url)

    pretrained_emb = np.load(npy_path_url)

    sentences = list()
    sentence_words = list()
    sentence_word_lengths = list()
    sentence_word_indices = list()

    claim_tok_list = []

    report_domains = []
    # source_vocab ids
    for ds in domain:
        # domain item to item id
        ds_item = torch.LongTensor([source_vocab[item] if item in source_vocab else unk_idx
                                        for item in ds]).to(device)
        report_domains.append(ds_item)
    report_domains = pad_sequence(report_domains, batch_first=True)
    # (batch_size, max_num_reports)

    # 合并每个claim下的所有句子
    for sents in _CLAIM_TOK:
        # 判断claim是否为多个句子组成的，如果是，再合并     
        if not isinstance(sents[0], list): 
            claim_tok_list = [tok for tok in _CLAIM_TOK] # keep consistant with reports
            break

        _sent = []
        for s in sents:
            _sent = _sent + s
            _sent = _sent[:512]
        claim_tok_list.append([_sent])

    # claim_len = [len(tok) for tok in claim_tok_list]
    # tgt_len = [len(tok) for tok in _TGT_TOK]
    # src_len = [len(tok) for tok in _SRC_TOK]
    # multi-doc sent_len
    src_sent_num = [[len(t) for t in tok] for tok in _SRC_TOK]# sents in each doc for a given claim
    src_doc_num = [sum(nums) for nums in src_sent_num] # total sent num in each doc for a given claim  

    src_len = []# length for each sent
    src_tok_list = []
    for tok in _SRC_TOK:
        for t in tok:
            src_tok_list.extend(t)
            src_len.extend([len(s) for s in t])

    # tok_list = claim_tok_list + _TGT_TOK + _SRC_TOK

    claim_tok_list = [[item] for item in claim_tok_list]
    src_tok_list = [[item] for item in src_tok_list]
    tok_list = claim_tok_list + src_tok_list #_SRC_TOK

    claim_len = [len(tok) for tok in claim_tok_list]
    # total sents in claim and src, respectively
    num_claim = sum([len(tok) for tok in claim_tok_list])
    # num_tgt = sum(tgt_len)
    num_src = sum(src_doc_num)
    # total_num = num_claim + num_tgt + num_src
    total_num = num_claim + num_src

    # # 得到pre+ N 个句子
    # c_sentences_index = get_idx_list(claim_len)
    # # t_sentences_index = get_idx_list(tgt_len)
    # s_sentences_index = [get_idx_list(s_num) for s_num in src_sent_num]#for each report in a given claim
    c_sentences_nums = claim_len
    s_sentences_nums = src_sent_num

    unk_idx = 1
    for sents in tok_list:
        for sent in sents:

            # word to word id
            sentence = torch.LongTensor([vocab[word] if word in vocab else unk_idx
                                            for word in sent]).to(device)

            # char of word to char id
            words = list()
            for word in sent:
                words.append([char_vocab[ch] if ch in char_vocab else unk_idx
                                for ch in word])

            # save word lengths
            word_lengths = torch.LongTensor([len(word) for word in words]).to(device)

            # sorting lengths according to length
            word_lengths, word_indices = torch.sort(word_lengths, descending=True)

            # sorting word according word length
            word_indices = word_indices.to(device)
            words = [torch.LongTensor(word).to(device) for word in words]

            # padding char tensor of words
            words = pad_sequence(words, batch_first=True).to(device)
            # (max_word_len, sent_len)

            sentences.append(sentence)
            sentence_words.append(words)
            sentence_word_lengths.append(word_lengths)
            sentence_word_indices.append(word_indices)

    # record sentence length and padding sentences
    sentence_lengths = [len(sentence) for sentence in sentences]
    # (batch_size)
    sentences = pad_sequence(sentences, batch_first=True)
    # (batch_size, max_sent_len)

    c_sentences             = sentences[:num_claim]
    c_sentence_lengths      = sentence_lengths[:num_claim]
    c_sentence_words        = sentence_words[:num_claim]
    c_sentence_word_lengths = sentence_word_lengths[:num_claim]
    c_sentence_word_indices = sentence_word_indices[:num_claim]
    # append indexs: c_sentences_nums
    c_results = (c_sentences, c_sentence_lengths, c_sentence_words, c_sentence_word_lengths, c_sentence_word_indices, c_sentences_nums)

    # t_sentences             = sentences[num_claim:num_claim+num_tgt]
    # t_sentence_lengths      = sentence_lengths[num_claim:num_claim+num_tgt]
    # t_sentence_words        = sentence_words[num_claim:num_claim+num_tgt]
    # t_sentence_word_lengths = sentence_word_lengths[num_claim:num_claim+num_tgt]
    # t_sentence_word_indices = sentence_word_indices[num_claim:num_claim+num_tgt]
    # # append t_sentences_index
    # t_results = (t_sentences, t_sentence_lengths, t_sentence_words, t_sentence_word_lengths, t_sentence_word_indices, t_sentences_index)

    s_sentences             = sentences[num_claim:total_num]
    s_sentence_lengths      = sentence_lengths[num_claim:total_num]
    s_sentence_words        = sentence_words[num_claim:total_num]
    s_sentence_word_lengths = sentence_word_lengths[num_claim:total_num]
    s_sentence_word_indices = sentence_word_indices[num_claim:total_num]
    # append s_sentences_nums
    s_results = (s_sentences, s_sentence_lengths, s_sentence_words, s_sentence_word_lengths, s_sentence_word_indices, s_sentences_nums)

    t_results = None
    #Furthermore, return claim_tok_list, src_tok_list 
    return c_results, t_results, s_results, claim_tok_list, src_tok_list, src_sent_num, report_domains

# def load_df_dataset(mode="test", filepath=ROOT_PROJ_PATH, batch_size=64, n_workers=2, shuffle=False):
#     '''mode = train, test, val'''
#     '''"..\\dataset\\oracles"'''
#     filename = pjoin(filepath, f'ruling_oracles_{mode}.tsv')
#     dataset = myDataset(filename)

#     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.my_collate, num_workers=n_workers)
#     # for idx, (a,b,c) in enumerate(train_loader):
#     #     print(idx, a,b,c)
#     return train_loader
def read_from_dir(path):
    ''''return all filenames in dir'''
    # 得到文件夹下所有文件的名称
    filenames = os.listdir(path)
    name_list = []
    for name in filenames:
        if '.json' in name: 
            name_list.append(name)
    return name_list


def load_df_dataset(mode, lm_emb, filepath=ROOT_PROJ_PATH, batch_size=64, n_workers=2, shuffle=False):
    '''mode = train, test, val'''
    '''MODE/1234.json corresponding to an instance'''
    root_path = pjoin(filepath, f'{mode}')# 模拟train.py输入
    # files = read_from_dir(root_path)#   
    # all_data = []
    # for file in files:
    #     filename = pjoin(root_path, file)# root/xxxx.json
    #     with open(filename, 'r', encoding='utf-8') as json_file:
    #         obj = json.load(json_file)
    #         all_data.append(obj)
    # dataset = myDataset(all_data, None)

    dataset = myDataset(root_path, lm_emb)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.my_collate, num_workers=n_workers)
    return train_loader

if __name__ == '__main__':
    # train_dataset = load_df_dataset("train")
    # val_dataset = load_df_dataset("val")
    torch.multiprocessing.set_start_method('spawn')

    from distilbert_emb import DistilEmbeddings, list2str
    lm_emb = DistilEmbeddings()
    data_loader = load_df_dataset("test", lm_emb)
    for idx, (claim_tensors, just_tensors, src_tensors, oracle_ids, label_ids) in enumerate(data_loader):
        print(idx, claim_tensors, just_tensors, src_tensors, oracle_ids, label_ids)