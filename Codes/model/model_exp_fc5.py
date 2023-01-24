# coding: utf-8
# based on fc4 - DistilBERT //////// docs 
# coding: utf-8
# 221030
import os
import sys
# from typing_extensions import Required
from numpy.core.fromnumeric import shape
from numpy.core.numeric import True_
from rouge_score.scoring import Score
from torch._C import device
from torch.nn.modules import padding
sys.path.append('helpers/')
sys.path.append('model/')
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
import math
# from helpers.lstm_attention import multihead_attention
import math
import json
from os.path import join as pjoin
# from helpers.lm_embeddings import lmembed

from transformers import DistilBertModel
# from transformers import RobertaConfig, RobertaModel
# from transformers import AlbertModel

# LABEL_IDS = {"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5}
# TOP_K = 4 # 4 for liar, 5/5 for PUBHEALTH and SEEFACT # choose top-k sents as evidences for enhancement!!! 
# TOP_N_DOC = 5 # 选前5个report 做为每个claim的依据 6选5
TOP_N_DOC = 18 # 选前5个report 做为每个claim的依据 30 选 18
MAX_ORACLE= 55 # 最多的oracle的句子数

# TOP_N_DOC = 1
# MAX_ORACLE= 12 # 最多的oracle的句子数
# TOP_N_DOC = 12 # avg liar_raw
# MAX_ORACLE= 30 # 最多的oracle的句子数 59/2

class ExplainFC(nn.Module):
    def __init__(self, hidden_size, n_tags, embedding_url=None, bidirectional=True, lstm_layers=1,
                 n_embeddings=None, embedding_dim=None, lm_embedding_dim = 76800, freeze=False, char_feat_dim=0, max_doc_num=12, vocab_article_source=None, source_dim=20, bert_model_or_path='distilbert-base-uncased'):
        super(ExplainFC, self).__init__()
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.max_doc_num = max_doc_num
        # self.config = RobertaConfig.from_pretrained(bert_model_or_path)
        self.bert_embedding = DistilBertModel.from_pretrained(bert_model_or_path).to(self.device)##'distilbert-base-uncased'
        # self.bert_embedding = RobertaModel.from_pretrained(bert_model_or_path, config=self.config).to(self.device)##may cause OOV   'distilroberta-base'

        # if embedding_url:
        #     self.embedding = nn.Embedding.from_pretrained(
        #         embeddings=torch.Tensor(np.load(embedding_url)),
        #         freeze=freeze
        #     )
        # else:
        #     self.embedding = nn.Embedding(n_embeddings, embedding_dim, padding_idx=0)#n_embeddings=93000
        
        self.source_dim = source_dim
        self.vocab_article_source = vocab_article_source
#         if self.vocab_article_source:
#             self.article_source_dict = json.load(open(self.vocab_article_source, "r", encoding="utf-8"))
#             self.article_source_embedding = nn.Embedding(len(self.article_source_dict), self.source_dim, padding_idx=0)
            
        # self.sent_embed = lmembed
        # self.lm_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # self.lm_model = BertModel.from_pretrained('bert-base-uncased')
        # self.lm_model = AlbertModel.from_pretrained('albert-base-v2')

        self.lm_embedding_dim = lm_embedding_dim# for sents
        self.embedding_dim = lm_embedding_dim#self.embedding.embedding_dim
        self.char_feat_dim = char_feat_dim
        self.word_repr_dim = self.embedding_dim + self.char_feat_dim


        self.dropout_claim = nn.Dropout(p=0.4)#0.4
        self.dropout_just = nn.Dropout(p=0.4)
        self.dropout_src = nn.Dropout(p=0.4)

        self.dropout_layer = nn.Dropout(p=0.3)
        
        self.norm = nn.LayerNorm(self.word_repr_dim)

        self.lstm = nn.LSTM(
            input_size=self.word_repr_dim,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.lstm_layers = lstm_layers
        self.n_tags = n_tags
        self.n_hidden = (1 + bidirectional) * hidden_size

        self.linear = nn.Linear(self.word_repr_dim, self.n_hidden)

        
        self.word_lstm = nn.LSTM(
            input_size=self.n_hidden,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=lstm_layers,
            batch_first=True
        )

        # LSTM for summarization
        self.sentence_lstm = nn.LSTM(
            input_size=self.n_hidden,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # LSTM for veracity prediction
        self.doc_lstm = nn.LSTM(
            input_size=self.n_hidden,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=lstm_layers,
            batch_first=True
        )
        # coherence-based evidence att
        self.biaffine_w = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.biaffine_b = nn.Linear(self.n_hidden, 1, bias=False)
        self.gate_w = nn.Linear(self.n_hidden, 1, bias=False)
        self.gate_b = nn.Linear(self.n_hidden, 1, bias=False)


        self.params = nn.ParameterDict({
            # 'doc_att_w': nn.Parameter(torch.rand(size=(self.n_hidden, self.n_hidden), requires_grad=True)),
            # 'bert_doc_att_w': nn.Parameter(torch.FloatTensor(self.lm_embedding_dim, self.lm_embedding_dim).uniform_(0, 0.2) , requires_grad=True),
            'doc_att_w': nn.Parameter(torch.FloatTensor(self.n_hidden, self.n_hidden).uniform_(0, 0.2) , requires_grad=True),
            'doc_sent_att_w': nn.Parameter(torch.FloatTensor(self.n_hidden, self.n_hidden).uniform_(0, 0.2) , requires_grad=True),
            'doc_sent_att_w2': nn.Parameter(torch.FloatTensor(self.n_hidden, self.n_hidden).uniform_(0, 0.2) , requires_grad=True),
            'doc_sent_att_w3': nn.Parameter(torch.FloatTensor(self.n_hidden, self.n_hidden).uniform_(0, 0.2) , requires_grad=True),
            'doc_sent_att_b': nn.Parameter(torch.FloatTensor(self.n_hidden).uniform_(0, 0.2) , requires_grad=True),
            # 'doc_sent_att_b2': nn.Parameter(torch.FloatTensor(self.n_hidden, 1).uniform_(0, 0.2) , requires_grad=True),
            # 'doc_sent_att_w': nn.Parameter(torch.rand(size=(self.n_hidden, self.n_hidden), requires_grad=True)),
            # 'doc_sent_att_w2': nn.Parameter(torch.rand(size=(self.n_hidden, self.n_hidden), requires_grad=True)),
            # 'doc_sent_att_w3': nn.Parameter(torch.rand(size=(self.n_hidden, self.n_hidden), requires_grad=True)),
            'src_claim_att': nn.Parameter(torch.FloatTensor(self.n_hidden, self.n_hidden).uniform_(0, 0.2) , requires_grad=True),
            'evi_claim_att': nn.Parameter(torch.FloatTensor(self.n_hidden, self.n_hidden).uniform_(0, 0.2) , requires_grad=True),

            'biaffine_w': nn.Parameter(torch.rand(size=(self.n_hidden, self.n_hidden), requires_grad=True)),
            'biaffine_b': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'gate_w': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'gate_w2': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'gate2_w': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'gate2_w2': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'evi_biaffine_w': nn.Parameter(torch.rand(size=(self.n_hidden, self.n_hidden), requires_grad=True)),
            'evi_biaffine_b': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'evi_biaffine_w2': nn.Parameter(torch.rand(size=(self.n_hidden, self.n_hidden), requires_grad=True)),##
            'evi_biaffine_b2': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'evi_gate_w': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'evi_gate_w2': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            # 'evi_gate_b': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'evi_gate_b': nn.Parameter(torch.FloatTensor(self.n_hidden, 1).uniform_(0, 0.2) , requires_grad=True),
            'tp_gate_w': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'tp_gate_w2': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'tp_gate_b': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'content_w': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'salience_w': nn.Parameter(torch.rand(size=(self.n_hidden, self.n_hidden), requires_grad=True)),
            'novelty_w': nn.Parameter(torch.rand(size=(self.n_hidden, self.n_hidden), requires_grad=True)),
            'position_w': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),
            'bias_b': nn.Parameter(torch.rand(size=(self.n_hidden, 1), requires_grad=True)),            
        })
        self.dropout_evi = nn.Dropout(p=0.3)
        #

        self.c_reduce_dim = nn.Linear(self.word_repr_dim + self.lm_embedding_dim, self.word_repr_dim)
        # self.init_linear(self.c_reduce_dim)
        self.s_reduce_dim = nn.Linear(self.word_repr_dim + self.lm_embedding_dim, self.word_repr_dim)
        # self.init_linear(self.s_reduce_dim)
        self.combine_layer = nn.Linear(self.word_repr_dim * 2, self.word_repr_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)

        self.rich_repr = nn.Linear(self.n_hidden * 3, self.n_hidden)
        # self.nli_layer = nn.Linear(self.n_hidden * 3, self.n_hidden)
        self.renorm = nn.LayerNorm(self.n_hidden*2)

        # self.doc_att = AttentionLayer(self.n_hidden, self.n_hidden, self.n_hidden)

        self.atten_layer = AttentionLayer(self.n_hidden, self.n_hidden, self.n_hidden)
        self.atten_layer2 = AttentionLayer(self.n_hidden, self.n_hidden, self.n_hidden)

        self.doc_linear = nn.Linear(self.n_hidden*3, 1)

        
        self.batch_matt = AttentionLayer(self.word_repr_dim, self.word_repr_dim, self.word_repr_dim)

        self.reduce_layer = nn.Linear(self.n_hidden, 1)

        
        self.classifier3 = FeedForward(input_dim=self.n_hidden*3, hidden_size=self.n_hidden, num_classes=n_tags, dropout_rate=0.2)#nn.Linear(self.n_hidden, n_tags) dropout_rate=0.4
        # self.classifier = FeedForward(self.n_hidden, self.n_hidden//2, num_classes=n_tags, dropout_rate=0.4)#nn.Linear(self.n_hidden, n_tags)
        self.mlp_layer = FeedForward(self.n_hidden*4, self.n_hidden, num_classes=1)
        # self.mlp_layer = nn.Linear(self.n_hidden, 1)
        self.maxpool1d = nn.AdaptiveMaxPool1d(1)
        self.linear_layer = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Tanh()
        )

        self.content_layer = nn.Linear(self.n_hidden, 1)

    def gen_batch_info(self, word_repr, num_sents, sent_lengths):
        '''
        :param src_tensors: claim_sent, claim_sent_lengths, claim_sent_words, claim_sent_word_lengths, claim_sent_word_indices, claim_sent_nums
        or src_tensors: just_sent, just_sent_lengths, just_sent_words, just_sent_word_lengths, just_sent_word_indices, just_sent_index
        or src_tensors: src_sent, src_sent_lengths, src_sent_words, src_sent_word_lengths, src_sent_word_indices, src_sent_nums
        :return:
        '''



        _idx = 0
        max_len = 0
        sents = []
        sent_lens = []
        # num_sents = []

        # obtain batch 
        batch_nums = [sum(s) for s in num_sents]
        for num in batch_nums:
            a = _idx
            b = a + num
            sents.append(word_repr[a:b])
            sent_lens.append(sent_lengths[a:b])
            if max_len < num:
                max_len = num
            _idx = b

        return sents, sent_lens, num_sents

    def extract_evidences_doc(self, claim_repr, batch_word_repr, doc_num_sents, sent_lengths, mask_mat, num_oracle_eachdoc, report_domains=None):
        '''
        :param
        claim_repr: claim tensor (batch_size, n_words, n_hidden)
        word_repr : src tensor, full text. (2000, n_words, n_hidden)
        
        sent_lengths: lengths of src
        mask_mat: batch_size groups of mask tensors.
        num_oracle_eachdoc: 关于一个claim， 每个相关的report 包含的oracle句子数量, (batch_size, 30)
        :return: evidence indices list.
        '''
        
        
        evi_logits = []
        
        evi_scores = []
        s_index = []#5, 10, 15, ...
        s_lengths = []
        _idx = 0

        selected_ids = []

        # totoal_sents = 0
        evi_masks = []# how many real sents

        # obtaining real length of each sent
        start = 0
        evi_sent_nums = [] #每个claim 包含多少个证据句子
        all_sent_lengths = [[] for _ in doc_num_sents]
        for i,num_list in enumerate(doc_num_sents):
            for num in num_list:
                all_sent_lengths[i].append(sent_lengths[start:start+num])
                start += num

        # (batch_size, 30, max_num_sent, hidden_size) 对每个batch中的30个报告doc进行处理
        # doc_num_sents ： 改成每个文档的句子数
        batch_size = batch_word_repr.shape[0]
        evi_sent_repr = [[] for _ in range(batch_size)]
        sort_indices_list = []# for choose docs

        selected_doc_reprs = []
        domain_report_reprs = []
        selected_doc_sent_reprs = []
        selected_doc_sent_masks = []
        selected_doc_sent_lengths = []

        evi_logits = [[] for _ in range(batch_size)]
        s_lengths = [[] for _ in range(batch_size)]
        evi_sents = [[] for _ in range(batch_size)]
        # evi_nums_each_doc = []
        truth_n_sents = []
        for i, (n_sents, n_sents_lens, claim, word_repr, mask, num_oracle, domains) in enumerate(zip(doc_num_sents, all_sent_lengths, claim_repr, batch_word_repr, mask_mat, num_oracle_eachdoc, report_domains), 0):
            '''For each claim, how to generate explanations from 30 reports, respectively'''
            a = _idx
            # word_repr[a:b] (n_sents, n_words, n_hidden): construct a batch of n comment sents in a claim.  
            # b =  a + n_sents#n_sents = b-a#word_repr[a:b].shape[0]

            # 1 doc repr (30, max_num_sent, 400) -> (30, 1, 400)
            max_num_sent = word_repr.shape[1]
            # real_num_sents = n_sents + [0 for _ in range(word_repr.shape[0]-len(n_sents))]

            s_packed = nn.utils.rnn.pack_padded_sequence(word_repr[:len(n_sents)], n_sents, batch_first=True, enforce_sorted=False).to(self.device)
            s_out, (s_hn, _) = self.sentence_lstm(s_packed)
            s_unpacked, _ = nn.utils.rnn.pad_packed_sequence(s_out, total_length=max_num_sent, batch_first=True)
        
            # doc-level repr: obtain the max pooling results 
            s_unpacked = torch.max(s_unpacked, dim=1)[0]

            # 2 doc attention for selection
            doc_scores = torch.matmul(torch.matmul(s_unpacked, self.params['doc_att_w']), claim) #N_doc
            # Ranking doc
            sorted_score, indices = torch.sort(doc_scores, descending=True)

            _doc_reprs = []
            _domain = []
            _doc_sent_reprs = []
            _doc_sent_masks = []
            _doc_sent_lengths = []
            # select TOP_N_DOC
            for j in indices.tolist()[:TOP_N_DOC]:
                _doc_reprs.append(s_unpacked[j]) # 文档级表示 TOP_N_DOC, (400)
                _doc_sent_reprs.append(word_repr[j]) # 文档句子级表示 TOP_N_DOC, (max_sent, 400)
                _doc_sent_masks.append(mask[j]) # TOP_N_DOC, (max_sent) ##, 400)
                _doc_sent_lengths.append(n_sents_lens[j]) # 文档句子长度 TOP_N_DOC, max_sent
                # _evi_sent_nums.append(n_sents[j]) # 每个文档包含多少个句子！
                _domain.append(domains[j])

            
            selected_doc_reprs.append(_doc_reprs)
            domain_report_reprs.append(_domain)
            selected_doc_sent_reprs.append(_doc_sent_reprs)
            selected_doc_sent_masks.append(_doc_sent_masks)
            selected_doc_sent_lengths.append(_doc_sent_lengths)
            # evi_sent_nums.append(_evi_sent_nums)

            sort_indices_list.append(indices.tolist()[:TOP_N_DOC])
            # 3 select sentences for explanation, min_oracle, threshold_oracle, max_oracle
            # MIN_ORACLE, THRESHOLD_ORACLE, MAX_ORACLE= 1, 18, 55#(或者report的最大长度)
            totoal_sents = 0
            _evi_sent_nums = []

            for doc_repr, doc_domain, doc_sent_reprs,doc_sent_masks,real_lens in zip(_doc_reprs, _domain, _doc_sent_reprs, _doc_sent_masks, _doc_sent_lengths):
                '''doc(max_sent, 400) claim(400)  doc_sent_att_w'''

                doc_sent_masks = 1 - doc_sent_masks
                # 1) claim-relevance, topic
                claim_scores = torch.matmul(torch.matmul(doc_sent_reprs.masked_fill(doc_sent_masks.unsqueeze(1) == 1, -1e9), self.params['doc_sent_att_w']), claim) / math.sqrt(doc_sent_reprs.shape[-1])#N_doc
                # sent_scores = torch.softmax(sent_scores, dim=-1)
                # if doc_sent_masks is not None:
                #     claim_scores = claim_scores.masked_fill(doc_sent_masks == 1, -1e9)
                claim_att_weights = F.softmax(claim_scores, dim=-1) # weights for each sent in a report doc

                # 2) entire doc-relevance, sailence
                doc_scores = torch.matmul(torch.matmul(doc_sent_reprs.masked_fill(doc_sent_masks.unsqueeze(1) == 1, -1e9), self.params['doc_sent_att_w2']), doc_repr) / math.sqrt(doc_sent_reprs.shape[-1])#N_doc
                doc_att_weights = F.softmax(doc_scores, dim=-1) # weights for each sent in a report doc

                # 3) content Wh ; doc_sent_att_w3
                # content_scores = torch.matmul(doc_sent_reprs.masked_fill(doc_sent_masks == 1, -1e9), self.params['doc_sent_att_b']) / math.sqrt(doc_sent_reprs.shape[-1])#N_doc
                content_scores = self.content_layer(doc_sent_reprs.masked_fill(doc_sent_masks.unsqueeze(1) == 1, -1e9)).squeeze() / math.sqrt(doc_sent_reprs.shape[-1])#N_doc
                content_att_weights = F.softmax(content_scores, dim=-1) # weights for each sent in a report doc

                # # 4) novelty: s_i and all - s_i
                # ones_masks = torch.ones(doc_sent_reprs.shape[1], doc_sent_reprs.shape[1]).to(self.device)
                # dynamic_masks = ones_masks - torch.diag_embed(torch.diag(ones_masks))
                # # dynamic_masks (n_sents, n_sents)
                # h_reduncy = torch.matmul(dynamic_masks, doc_sent_reprs)/(max_num_sent-1.0)
                # 5) bias term

                total_weights = claim_att_weights+doc_att_weights+content_att_weights
                pred_ids = self.sigmoid(total_weights.masked_fill(doc_sent_masks == 1, -1e9))
                evi_logits[i].append(pred_ids)
                # Ranking
                sorted_total_score, sorted_total_indices = torch.sort(pred_ids, descending=True)

                ind_list = []
                for score,ind,_mask in zip(sorted_total_score, sorted_total_indices,doc_sent_masks):
                    if score > 0.505:
                        # total_weights[ind]
                        ind_list.append(ind)

                # _end = max_num_sent if  max_num_sent < TOP_N_DOC else TOP_N_DOC
                # _end = _end if _end < MAX_ORACLE else MAX_ORACLE
                _end = min([len(ind_list), MAX_ORACLE])
                truth_n_sents.append(_end)# collect how many sents in each doc
                _evi_sent_nums.append(_end) # 每个文档包含多少个证据！
                
                evi_repr, _ = self.atten_layer2(doc_sent_reprs, doc_sent_reprs, doc_sent_reprs, mask=doc_sent_masks.unsqueeze(-1))

                # select Top - 4/5/5 sents in accord with the number of oracle ids.
                selected_sent_repr = torch.vstack([evi_repr[ind] for ind in ind_list[:_end]] )
                # sent reprs
                evi_sents[i].append(selected_sent_repr)
                # _evi_sent_nums.append(selected_sent_repr.shape[0]) # 每个文档包含多少个证据！

                # sorted_score = F.softmax(sorted_score)
                # evi_sents.append(selected_sent_repr * ((sorted_score[indices[:_end]]).unsqueeze(-1)).unsqueeze(-1))
                selected_ids.append(ind_list[:_end])

                # evi_logits.append(beta * sorted_score)

                # gen sent_index and sent_lengths for gen_batch_info
                s_index.append(totoal_sents + _end)
                # s_lengths.extend(sent_lengths[a:b][:_end])
                # using the sorted lengths
                # s_lengths.extend([sent_lengths[a:a+max_num_sent][i] for i in ind_list[:_end]])
                s_lengths[i].extend([real_lens[i] for i in ind_list[:_end]])

                totoal_sents = totoal_sents + _end
                _idx += max_num_sent

            evi_sent_nums.append(_evi_sent_nums)              

        # evi_lens = [len(item) for item in evi_sents]
        # return repr and selected ids.
        
        # abs difference
        # abs_repr = torch.vstack(abs_repr)

        evi_sents  = [torch.vstack(item) for item in evi_sents] # (batch_size, n_sents, hidden_size)
        # truth_n_sents = [len(item) for item in evi_sents] #lens for evi_logits

        evi_sents = pad_sequence(evi_sents, batch_first=True) #torch.vstack(evi_sents)


        # evi_logits = torch.vstack(evi_logits) #(TOP_N_DOC, 31)  pad_sequence(evi_logits, batch_first=True)
        evi_logits = pad_sequence([torch.vstack(items) for items in evi_logits], batch_first=True) #(batch_size, TOP_N_DOC, 31)  pad_sequence(evi_logits, batch_first=True)
        evi_logits_masks = pad_sequence([torch.vstack(item) for item in selected_doc_sent_masks], batch_first=True)
        # (batch_size, max_n_sents)
        # evi_masks = torch.FloatTensor([[True]*t + [False]*(evi_sents.shape[1] - t) for t in truth_n_sents]).to(self.device)
        # doc_n_sents = [sum(item) for item in evi_sent_nums]
        # evi_masks = torch.FloatTensor([[True]*t + [False]*(evi_sents.shape[1] - t) for t in doc_n_sents]).to(self.device)
        evi_masks = []
        for doc_n_sents in evi_sent_nums:
            # 每个claim
            _masks = torch.FloatTensor([[True]*t + [False]*(evi_sents.shape[1] - t) for t in doc_n_sents]).to(self.device)
            evi_masks.append(_masks)

        evi_masks = pad_sequence(evi_masks, batch_first=True)
        # return evi_sents, evi_logits, selected_ids, evi_masks, s_index, s_lengths, selected_doc_reprs
        # sort_indices_list collecting indices.tolist()[:TOP_N_DOC]
        # return evi_sents, evi_logits, sort_indices_list, selected_ids, evi_masks, s_index, s_lengths, selected_doc_reprs
        return evi_sents, (evi_logits, evi_logits_masks), sort_indices_list, selected_ids, evi_masks, evi_sent_nums, s_lengths, selected_doc_reprs

    def forward(self, oracle_ids, labels=None, lm_ids_dict=None):
    # def forward(self, claim_tensors, just_tensors, src_tensors, oracle_ids, labels=None, lm_ids_dict=None):
        ''' Task 1: Veracity Prediction -- T/F/H...
            1) claim + src --> veracity
            2) claim + src[oracle_ids] --> veracity
            3) claim + just --> veracity
            4)*claim + src + just --> veracity
            5)*claim + src[oracle_ids] + just --> veracity

            Task 2: Sentence Extraction -- Explanation 1/0
            1) src --> src[oracle_ids]

        claim_tensors:
        just_tensors: human writen justification
        src_tensors: original text tensor
        '''        
        claim_ids, claim_attention_mask = lm_ids_dict['claim_ids'], lm_ids_dict['claim_masks']
        src_ids, src_attention_mask = lm_ids_dict['src_ids'], lm_ids_dict['src_masks']
        batch_size = len(claim_ids)
        claim_repr = [self.bert_embedding(_claim.to(self.device)).last_hidden_state[:,0,:] for _claim in claim_ids]
        src_repr = []
        # [self.bert_embedding(_src.to(self.device)).last_hidden_state for _src in src_ids]
        for _src in src_ids:
            src_repr.append([self.bert_embedding(s.to(self.device)).last_hidden_state[:,0,:] for s in _src])

        claim_sent_repr = claim_repr #pad_sequence(claim_repr, batch_first=True) # (batch_size, 1, bert_dim)
        src_sent_repr = [pad_sequence(src, batch_first=True) for src in src_repr] # (batch_size, 12, bert_dim)
    

        # 文档包含的真实句子数量 按照文档分开
        src_sent_num = lm_ids_dict['src_sent_num'] # sents in each doc for a given claim
        src_doc_num = [sum(nums) for nums in src_sent_num] # total sent num in each doc for a given claim         

        # for each claim: if batch>1
        selected_s_repr = []
        selected_indices_list = []
        src_mask = []

        selected_ids = [[] for _ in range(batch_size)]
        evi_logits = [[] for _ in range(batch_size)]
        s_lengths = [[] for _ in range(batch_size)]
        evi_sents = [[] for _ in range(batch_size)]
        batch_sel_thresholds = [[] for _ in range(batch_size)]
        selected_sent_repr = []
        selected_sent_repr_mask = []

        veracity = []
        
        for i, (pad_c_repr, pad_s_repr, real_num_src) in enumerate(zip(claim_sent_repr, src_sent_repr, src_sent_num)):
            '''pad_c_repr(1, 768); pad_s_repr(12, xx, 768)--这里的doc_num 12可以暂时看作batch_size'''
            # For 每个claim; 循环batch_size次
            _mask = torch.FloatTensor([[True]*t + [False]*(pad_s_repr.shape[1] - t) for t in real_num_src]).to(self.device)
            src_mask.append(_mask)

            # 1: extract report docs
            # doc repr
            s_packed = nn.utils.rnn.pack_padded_sequence(pad_s_repr, real_num_src, batch_first=True, enforce_sorted=False).to(self.device)
            s_out, (s_hn, _) = self.lstm(s_packed)
            # hn (n_layers * n_directions, batch_size, hidden_size)

            s_unpacked, _ = nn.utils.rnn.pad_packed_sequence(s_out, total_length=max(real_num_src), batch_first=True)
            # s_unpacked (batch_size, max_sent_len, n_hidden) 12, 68, 768

            # s_unpacked, p_score = self.atten_layer(s_unpacked, s_unpacked, s_unpacked, mask=(_mask==False).unsqueeze(-1))#test!!!!!!!!!!!!!!!!!
            # doc-level repr: obtain the max pooling results 
            s_unpacked = torch.max(s_unpacked, dim=1)[0]

            # doc attention for selection
            doc_scores = torch.matmul(torch.matmul(s_unpacked, self.params['doc_att_w']), pad_c_repr.squeeze(0)) #N_doc
            # Ranking doc
            sorted_score, indices = torch.sort(doc_scores, descending=True)
            _indices = indices.tolist()[:TOP_N_DOC]
            # selected doc indices
            selected_indices_list.append(_indices)
            # selected doc reprs 缩小范围 coarse to fine
            # sel_s_repr = torch.vstack([pad_s_repr[i] for i in _indices]) # (TOP_N_DOC*max_num, 768)
            sel_s_repr = pad_sequence([pad_s_repr[i] for i in _indices], batch_first=True) # (TOP_N_DOC, max_num, 768)
            sel_s_mask = pad_sequence([_mask[i] for i in _indices], batch_first=True) # (TOP_N_DOC, max_num)
            selected_sent_repr_mask.append(sel_s_mask)

            h_reduncy = torch.zeros(sel_s_repr.shape[-1]).to(self.device)

            # 2: select evidence sents: redundancy 循环TOP_N_DOC次
            for doc_repr, s_repr, s_mask in zip(s_unpacked, sel_s_repr, sel_s_mask):
                '''doc_repr(1,768)文档表示; pad_c_repr (1,768), s_repr(13, 768), s_mask (13)'''
                threshhold = 1.0/sum(s_mask) if sum(s_mask) > 1 else 1.0/2  # total_n_sents
                batch_sel_thresholds[i].append(threshhold)
                s_mask = 1 - s_mask
                # 1) claim-relevance, topic
                claim_scores = torch.matmul(torch.matmul(s_repr.masked_fill(s_mask.unsqueeze(1) == 1, -1e9), self.params['doc_sent_att_w']), pad_c_repr.squeeze(0)) / math.sqrt(s_repr.shape[-1])#N_doc
                # claim_att_weights = F.softmax(claim_scores, dim=-1) # weights for each sent in a report doc

                # 2) entire doc-relevance, sailence
                doc_scores = torch.matmul(torch.matmul(s_repr.masked_fill(s_mask.unsqueeze(1) == 1, -1e9), self.params['doc_sent_att_w2']), doc_repr) / math.sqrt(s_repr.shape[-1])#N_doc
                # doc_att_weights = F.softmax(doc_scores, dim=-1) # weights for each sent in a report doc

                # 3) content Wh ; doc_sent_att_w3
                content_scores = self.content_layer(s_repr.masked_fill(s_mask.unsqueeze(1) == 1, -1e9)).squeeze() / math.sqrt(s_repr.shape[-1])#N_doc
                # content_att_weights = F.softmax(content_scores, dim=-1) # weights for each sent in a report doc

                # 4) novelty
                red_scores = torch.matmul(torch.matmul(s_repr.masked_fill(s_mask.unsqueeze(1) == 1, -1e9), self.params['doc_sent_att_w3']), h_reduncy) / math.sqrt(s_repr.shape[-1])#N_doc
                # red_att_weights = F.softmax(red_scores, dim=-1) # weights for each sent in a report doc

                # total_weights = claim_att_weights+doc_att_weights #+content_att_weights
                total_weights = claim_scores + doc_scores + content_scores - red_scores #+content_att_weights
                pre_prob = self.sigmoid(total_weights) # .masked_fill(s_mask == 1, -1e9)
                # probability of choosing evidence for each batch
                evi_logits[i].append(pre_prob)

                # update previous state for next iteration!
                h_reduncy = self.tanh(torch.sum(s_repr * pre_prob.unsqueeze(-1), dim=0)) #[768]

                # h_reduncy = self.dropout_layer(h_reduncy)


                ind_list = []
                for ind,score in enumerate(pre_prob):
                    if score > threshhold:
                        ind_list.append(ind)
                # sorted_total_score, sorted_total_indices = torch.sort(pre_prob, descending=True)# Ranking xxx
                # for score,ind in zip(sorted_total_score, sorted_total_indices):
                #     if score > 1.0/total_n_sents:#0.505:
                #         ind_list.append(ind)


                # avoiding selecting zero item, 既然选择该文档，说明至少存在一个句子是证据
                if len(ind_list) == 0:
                    ind_list.append(0)

                # 3：obtain refined sent reprs
                _end = min([len(ind_list), MAX_ORACLE])                
                # evi_repr, _ = self.atten_layer2(s_repr, s_repr, s_repr, mask=s_mask)
                evi_repr = s_repr

                # select Top - 4/5/5 sents in accord with the number of oracle ids.
                # selected_sent_repr = torch.vstack([evi_repr[ind] for ind in ind_list[:_end]] )
                for ind in ind_list[:_end]:
                    selected_sent_repr.append(evi_repr[ind])
                    # selected_sent_repr_mask.append((1-s_mask)[ind])
                # sent reprs
                # evi_sents[i].append(evi_repr)
                # _evi_sent_nums.append(selected_sent_repr.shape[0]) # 每个文档包含多少个证据！
                selected_ids[i].append(ind_list[:_end])

                # return selected_indices_list, selected_ids/evi_logits, selected_sent_repr
            
            # Task: doc clf 
            # return selected_indices_list  for doc clf (batch_size, 1)

            # 4: veracity prediction
            doc_sent_repr = torch.vstack(selected_sent_repr) 
            pool_sent_repr = torch.max(doc_sent_repr, dim=0, keepdim=True)[0]
            pool_src_repr = torch.max(s_unpacked, dim=0, keepdim=True)[0]

            rich_repr = torch.cat([pad_c_repr, self.dropout_layer(pool_sent_repr), self.dropout_layer(pool_src_repr)], dim=-1)

            # task 2
            ver = self.classifier3(rich_repr)
            veracity.append(ver)

        veracity = torch.vstack(veracity)
        evi_logits = [torch.vstack(evi) for evi in evi_logits]
        # return veracity, doc-clf, sent-clf 
        return veracity, selected_indices_list, (evi_logits, selected_ids, selected_sent_repr_mask, batch_sel_thresholds)

class AttentionLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim):
        super(AttentionLayer, self).__init__()
        self.linear_q = nn.Linear(q_dim, q_dim)
        self.linear_k = nn.Linear(k_dim, k_dim)
        self.linear_v = nn.Linear(v_dim, v_dim)

    def forward(self, query, key, value, mask=None, norm=True, dropout=None):
        # compute scaled dot prodcut attention
        d_k = query.size(-1)
        scores = torch.matmul(self.linear_q(query), self.linear_k(key).transpose(-2, -1))
        if norm:
            scores = scores / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = F.dropout(p_attn, p=dropout)
        return torch.matmul(p_attn, self.linear_v(value)), p_attn

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_size=100, dropout_rate=0., num_classes=3):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return out







