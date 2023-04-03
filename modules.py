#!/usr/bin/env python
# coding=utf-8
"""
model: span-level and seq-labeled ner model
started from 2021/1
"""
import math, time, os
from typing import *
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, RobertaModel, AutoModel, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, AutoConfig
import ipdb
from datautils import Meaner, CUDA_Recorder, Time_Recorder
import logging

logger = logging.getLogger(__name__)
#
# import taichi as ti
# ti.init(arch=ti.gpu)

cudar1 = CUDA_Recorder(torch)
cudar2 = CUDA_Recorder(torch)
time1 = Time_Recorder()
time2 = Time_Recorder()

cudar1.surpress = True
cudar2.surpress = True
time1.surpress = True
time2.surpress = True


def count_params(model_or_params: Union[torch.nn.Module, torch.nn.Parameter, List[torch.nn.Parameter]],
                 return_trainable=True, verbose=True):
    """
    NOTE: `nn.Module.parameters()` return a `Generator` which can only been iterated ONCE.
    Hence, `model_or_params` passed-in must be a `List` of parameters (which can be iterated multiple times).
    """
    if isinstance(model_or_params, torch.nn.Module):
        model_or_params = list(model_or_params.parameters())
    elif isinstance(model_or_params, torch.nn.Parameter):
        model_or_params = [model_or_params]
    elif not isinstance(model_or_params, list):
        raise TypeError("`model_or_params` is neither a `torch.nn.Module` nor a list of `torch.nn.Parameter`, "
                        "`model_or_params` should NOT be a `Generator`. ")

    num_trainable = sum(p.numel() for p in model_or_params if p.requires_grad)
    num_frozen = sum(p.numel() for p in model_or_params if not p.requires_grad)

    if verbose:
        logger.info(f"The model has {num_trainable + num_frozen:,} parameters, "
                    f"in which {num_trainable:,} are trainable and {num_frozen:,} are frozen.")

    if return_trainable:
        return num_trainable
    else:
        return num_trainable + num_frozen


class Biaffine_layer(nn.Module):
    def __init__(self,
                 start_size,
                 end_size,
                 ent_size,
                 add_bias=True):
        super(Biaffine_layer, self).__init__()

        self.add_bias = add_bias

        self.w = nn.Parameter(torch.Tensor(ent_size, start_size + int(self.add_bias), end_size + int(self.add_bias)))
        self.w.data.zero_()
        # self.w.data.uniform_(-0.1, 0.1)

    def forward(self, start_hidden, end_hidden):
        # start_hidden [b,e,l,h]  # e = ent_size = head,  h = start hidden
        # end_hidden [b,e,l,h]
        # return span_ner_mat_tensor [b,l,l,e]
        if self.add_bias:  # 在hidden维补1
            start_hidden = torch.cat([start_hidden, torch.ones_like(start_hidden[..., :1])], dim=-1)
            end_hidden = torch.cat([end_hidden, torch.ones_like(end_hidden[..., :1])], dim=-1)

        # bilinear
        # w_shape: [e,i,j]
        # start_shape: [b,e,m,i]
        # end_shape: [b,e,n,j]
        # start*w*end -> b,m,n,e
        # res = torch.einsum('eij, benj -> beni', self.w, end_hidden)  # w * end
        # span_ner_mat_tensor = torch.einsum('bemi, beni -> bmne', start_hidden, res)  # w *t2
        span_ner_mat_tensor = torch.einsum('bemi,eij,benj->bmne', start_hidden, self.w, end_hidden)
        return span_ner_mat_tensor  # [b,l,l,e]

    def fast_forward(self, start_hidden, end_hidden):
        # 当传入的为筛选过的start,end
        # start_hidden [*,e,h]  # e = ent_size = head, h = start/end_hidden
        # end_hidden [*,e,h]
        # return span_ner_mat_tensor [*,e]
        if self.add_bias:  # 在hidden维补1
            start_hidden = torch.cat([start_hidden, torch.ones_like(start_hidden[..., :1])], dim=-1)
            end_hidden = torch.cat([end_hidden, torch.ones_like(end_hidden[..., :1])], dim=-1)

        # bilinear
        # w_shape: [e,i,j]  i,j->h
        # start_shape: [*,e,i]
        # end_shape: [*,e,j]
        # start*w*end -> *,e
        # res = torch.einsum('eij, cej -> cei', self.w, end_hidden)  # w * end
        # span_ner_mat_tensor = torch.einsum('cei, cei -> ce', start_hidden, res)  # w *t2
        span_ner_mat_tensor = torch.einsum('cei,eij,cej->ce', start_hidden, self.w, end_hidden)
        return span_ner_mat_tensor  # [*,e]


class Add_Attn_Layer(nn.Module):
    def __init__(self, start_size, end_size):
        super(Add_Attn_Layer, self).__init__()

        assert start_size == end_size
        self.v = nn.Parameter(torch.Tensor(start_size))
        self.v.data.uniform_(-0.1, 0.1)

    def forward(self, start_hidden, end_hidden):
        # start_hidden [b,e,l,h] end_hidden [b,e,l,h] e = ent_size = head,  h = start/end_hidden
        # return span_ner_mat_tensor [b,l,l,e]
        start_hidden = start_hidden[:, :, :, None, :]  # [b,e,l,1,h]
        end_hidden = end_hidden[:, :, None, :, :]  # [b,e,1,l,h]
        add_hidden = start_hidden + end_hidden  # [b,e,l,l,h]  # 利用维度1的广播
        add_hidden = torch.tanh(add_hidden)  # [b,e,l,l,h]

        res = torch.matmul(add_hidden, self.v)  # [b,e,l,l,h] * [h] = [b,e,l,l]
        span_ner_mat_tensor = res.permute(0, 2, 3, 1)  # [b,l,l,e]
        return span_ner_mat_tensor

    def fast_forward(self, start_hidden, end_hidden):
        # 当传入的为筛选过的start,end
        # start_hidden [*,e,h]  # e = ent_size = head, h = start/end_hidden
        # end_hidden [*,e,h]
        # return span_ner_mat_tensor [*,e]
        add_hidden = start_hidden + end_hidden  # [*,e,h]
        add_hidden = torch.tanh(add_hidden)  # [*,e,h]

        span_ner_mat_tensor = torch.matmul(add_hidden, self.v)  # [*,e,h] * [h] = [*,e]
        return span_ner_mat_tensor


class Simple_Concat_Layer(nn.Module):
    def __init__(self, start_size, end_size):
        super(Simple_Concat_Layer, self).__init__()

        assert start_size == end_size
        self.v = nn.Parameter(torch.Tensor(start_size + end_size))
        self.v.data.uniform_(-0.1, 0.1)

    def forward(self, start_hidden, end_hidden):
        # start_hidden [b,e,l,h] end_hidden [b,e,l,h] e = ent_size = head,  h = start/end_hidden
        # return span_ner_mat_tensor [b,l,l,e]
        length = start_hidden.shape[2]
        start_hidden = start_hidden[:, :, :, None, :].expand(-1, -1, -1, length, -1)  # [b,e,l,1,h]
        end_hidden = end_hidden[:, :, None, :, :].expand(-1, -1, length, -1, -1)  # [b,e,1,l,h]
        concat_hidden = torch.cat([start_hidden, end_hidden], dim=-1)  # [b,e,l,l,2h]  # 利用维度1的广播

        res = torch.matmul(concat_hidden, self.v)  # [b,e,l,l,2h] * [2h] = [b,e,l,l]

        span_ner_mat_tensor = res.permute(0, 2, 3, 1)  # [b,l,l,e]
        return span_ner_mat_tensor

    def fast_forward(self, start_hidden, end_hidden):
        # 当传入的为筛选过的start,end
        # start_hidden [*,e,h]  # e = ent_size = head, h = start/end_hidden
        # end_hidden [*,e,h]
        # return span_ner_mat_tensor [*,e]
        add_hidden = torch.cat([start_hidden, end_hidden], dim=-1)  # [*,e,2h]

        span_ner_mat_tensor = torch.matmul(add_hidden, self.v)  # [*,e,2h] * [2h] = [*,e]
        return span_ner_mat_tensor


class Complex_Concat_Layer(nn.Module):
    def __init__(self, start_size, end_size):
        super(Complex_Concat_Layer, self).__init__()

        assert start_size == end_size
        self.v = nn.Parameter(torch.Tensor(4 * start_size))
        self.v.data.uniform_(-0.1, 0.1)

    def forward(self, start_hidden, end_hidden):
        # start_hidden [b,e,l,h] end_hidden [b,e,l,h] e = ent_size = head,  h = start/end_hidden
        # return span_ner_mat_tensor [b,l,l,e]
        length = start_hidden.shape[2]
        start_hidden = start_hidden[:, :, :, None, :]  # [b,e,l,1,h]
        end_hidden = end_hidden[:, :, None, :, :]  # [b,e,1,l,h]
        sub_hidden = start_hidden - end_hidden  # [b,e,l,l,h]  # 利用维度1的广播
        mul_hidden = start_hidden * end_hidden  # [b,e,l,l,h]  # 利用维度1的广播

        start_hidden = start_hidden.expand(-1, -1, -1, length, -1)  # [b,e,l,1,h]
        end_hidden = end_hidden.expand(-1, -1, length, -1, -1)  # [b,e,1,l,h]
        concat_hidden = torch.cat([start_hidden, end_hidden, sub_hidden, mul_hidden], dim=-1)  # [b,e,l,l,2h]  # 利用维度1的广播

        res = torch.matmul(concat_hidden, self.v)  # [b,e,l,l,4h] * [4h] = [b,e,l,l]

        span_ner_mat_tensor = res.permute(0, 2, 3, 1)  # [b,l,l,e]
        return span_ner_mat_tensor

    def fast_forward(self, start_hidden, end_hidden):
        # 当传入的为筛选过的start,end
        # start_hidden [*,e,h]  # e = ent_size = head, h = start/end_hidden
        # end_hidden [*,e,h]
        # return span_ner_mat_tensor [*,e]
        sub_hidden = start_hidden - end_hidden  # [*,e,h]  # 利用维度1的广播
        mul_hidden = start_hidden * end_hidden  # [*,e,2h]  # 利用维度1的广播
        concat_hidden = torch.cat([start_hidden, end_hidden, sub_hidden, mul_hidden], dim=-1)  # [*,e,2h]

        span_ner_mat_tensor = torch.matmul(concat_hidden, self.v)  # [*,e,4h] * [4h] = [*,e]
        return span_ner_mat_tensor


def transpose_for_scores(x, num_heads, head_size, transopose=True):
    """ split head """
    # x: [bat,len,totalhid]
    new_x_shape = x.size()[:-1] + (num_heads, head_size)  # [bat,len,num_ent,hid]
    # new_x_shape = x.size()[:-1] + (self.num_ent*2 + 1, self.hidden_size_per_ent)  # [bat,len,num_ent,hid]
    x = x.view(*new_x_shape)
    if transopose:
        x = x.permute(0, 2, 1, 3)  # [bat,num_ent,len,hid]
    return x


def fast_decode_0120(threshold, batch_links, batch_lens, batch_start_hid, batch_end_hid):
    # print(batch_links.shape)
    # print(batch_lens.shape)
    # print(batch_start_hid.shape)
    # print(batch_end_hid.shape)
    # batch_links [b,l-1]
    # batch_lens [b]
    hidden_size_per_ent = batch_start_hid.shape[-1]
    ent_size = batch_start_hid.shape[1]

    # cudar2.reset()
    # time2.reset()
    result_matrix, indices, start_indices, end_indices = get_remained_indices_of_spans(batch_links > threshold, batch_lens)
    # print(start_indices)  # [bsz, *]
    # print(end_indices)  # [bsz,*]
    # print(indices)
    # time2.record('fast_decode get_remained_indices')
    # cudar2.record('fast_decode get_remained_indices')
    # total_indices  [3,*]
    # start_indices  [2,*]
    # end_indices  [2,*]
    # start_hid, end_hid [b,e,l,h]

    # memory_format=torch.legacy_contiguous_format
    # cudar2.reset()
    # time2.reset()
    # batch_start_hid = batch_start_hid.transpose(1, 2)  # [b,l,e,h]
    # batch_end_hid = batch_end_hid.transpose(1, 2)  # [b,l,e,h]
    # time2.record('fast_decode transpose')
    # cudar2.record('fast_decode transpose')

    # torch.cuda.synchronize()
    # cudar2.reset()
    # time2.reset()
    batch_start_hid1 = batch_start_hid[list(start_indices)]  # gather_nd [*,e,h]
    batch_end_hid1 = batch_end_hid[list(end_indices)]  # gather_nd [*,e,h]
    # time2.record('fast_decode gather_by_indices')
    # cudar2.record('fast_decode gather_by_indices')

    # 这种通过mask_select没有上面直接索引值这么快 有病吧 我下面自己实现了 重新做了自己做了的事麻痹
    # cudar2.reset()
    # time2.reset()
    # result_matrix, indices = get_remained_indices_of_spans(batch_links > threshold, batch_lens, flag=False)
    # length = batch_start_hid.shape[2]
    # num = indices.shape[1]
    # batch_start_hid = torch.masked_select(batch_start_hid.transpose(1, 2).unsqueeze(2).expand(-1,-1,length,-1,-1),
    #                                       result_matrix[...,None,None].bool()).view(num, 18, 50)
    # batch_end_hid = torch.masked_select(batch_end_hid.transpose(1, 2).unsqueeze(2).expand(-1,-1,length,-1,-1),
    #                                       result_matrix[...,None,None].bool()).view(num, 18, 50)
    # time2.record('fast_decode gather_by_indices')
    # cudar2.record('fast_decode gather_by_indices')

    # 这种通过mask_select没有上面直接索引值这么快
    # cudar2.reset()
    # time2.reset()
    # mask = result_matrix.bool()  # [b,l,l]
    # batch_start_hid = torch.masked_select(batch_start_hid[:,:,None,:,:], mask[:,:,:,None,None]).view(-1, ent_size, hidden_size_per_ent)
    # batch_end_hid = torch.masked_select(batch_end_hid[:,None,:,:,:], mask[:,:,:,None,None]).view(-1, ent_size, hidden_size_per_ent)
    # time2.record('fast_decode gather_by_indices')
    # cudar2.record('fast_decode gather_by_indices')

    return batch_start_hid1, batch_end_hid1, indices  # [*,e,h] [*,e,h] [3,*]


def calc_refined_mat_tensor(link_scores, pooling_type, temp=1):
    # link_scores [b,l-1]
    # span_ner_mat_tensor [b,l,l,e]
    if pooling_type == 'softmin':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='min', use_soft=True, temp=temp)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'min':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='min', use_soft=False)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'softmax':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='max', use_soft=True, temp=temp)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'max':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='max', use_soft=False)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'mean':
        mask_matrix = aggregate_mask_by_cum(link_scores, mean=True)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'sum':
        mask_matrix = aggregate_mask_by_cum(link_scores, mean=False)[..., None]  # b,l-1,l-1,1
    else:
        raise NotImplementedError
    final_mask = torch.nn.functional.pad(mask_matrix, pad=(0, 0, 1, 0, 0, 1), mode="constant", value=0)  # b,l,l,1  # 长宽增加1对齐

    return final_mask


class Bert_Span(nn.Module):
    def __init__(self, args):
        super(Bert_Span, self).__init__()
        self.char2id = args.datareader.char2id
        self.tag2id = args.datareader.tag2id
        self.ent2id = args.datareader.ent2id
        self.id2char = args.datareader.id2char
        self.id2tag = args.datareader.id2tag
        self.id2ent = args.datareader.id2ent
        self.args = args

        self.bert_conf = AutoConfig.from_pretrained(args.bert_model_dir)

        self.use_bilstm = False

        if args.pretrain_mode == 'fine_tuning':
            # bert_model_dir = 'hfl/chinese-bert-wwm-ext'
            # bert_model_dir = 'bert-base-chinese'
            if args.bert_model_dir.endswith('roberta-base'):
                self.bert_layer = AutoModel.from_pretrained(args.bert_model_dir,
                                                            hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2
                                                            )
            else:
                self.bert_layer = AutoModel.from_pretrained(args.bert_model_dir)
            if self.use_bilstm:
                self.bilstm_layer = nn.LSTM(
                    input_size=self.bert_conf.hidden_size,  # 768base 1024large
                    hidden_size=400 // 2,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0.
                )
            self.group_aggregating = SequenceGroupAggregating(mode='mean')

        if args.pretrain_mode == 'feature_based':
            self.bilstm_layer = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0
            )

        self.loss_type = args.span_loss_type
        self.interact_type = args.interact_type
        self.use_slr = args.use_slr
        self.pooling_type = args.pooling_type
        self.dropout_rate = args.dropout_rate

        self.vocab_size = len(self.char2id)
        # self.tag_size = len(self.tag2id)
        self.ent_size = len(self.ent2id)
        # self.dropout_layer = nn.Dropout(p=self.dropout_rate)  # TODO
        self.dropout_layer = nn.Dropout(p=args.dropout_rate)
        self.dropout_layer2 = nn.Dropout(p=0.2)
        self.dropout_layer5 = nn.Dropout(p=0.5)

        self.start_size = self.end_size = self.link_size = 50  # TODO

        total_ffn_size = self.ent_size * self.start_size * 2

        if self.use_slr:
            total_ffn_size += self.link_size * 2

        if self.use_bilstm:
            self.ffn_layer = nn.Linear(400, total_ffn_size)
        else:
            self.ffn_layer = nn.Linear(self.bert_conf.hidden_size, total_ffn_size)

        if self.interact_type == 'biaffine':
            self.interact_layer = Biaffine_layer(start_size=self.start_size, end_size=self.end_size, ent_size=self.ent_size)
        elif self.interact_type == 'self_attn':
            pass
        elif self.interact_type == 'add_attn':
            self.interact_layer = Add_Attn_Layer(start_size=self.start_size, end_size=self.end_size)
        elif self.interact_type == 'sconcat':
            self.interact_layer = Simple_Concat_Layer(start_size=self.start_size, end_size=self.end_size)
        elif self.interact_type == 'cconcat':
            self.interact_layer = Complex_Concat_Layer(start_size=self.start_size, end_size=self.end_size)
        else:
            raise NotImplementedError

        if self.loss_type == 'sigmoid':
            self.loss_layer = nn.BCEWithLogitsLoss(reduction='none')
        if self.loss_type == 'softmax':
            self.loss_layer = nn.CrossEntropyLoss(reduction='none')
        self.link_loss_layer = nn.BCEWithLogitsLoss(reduction='none')
        # print(*[n for n, p in self.named_parameters()], sep='\n')
        count_params(self)

        if self.args.pretrain_mode == 'fine_tuning':
            no_decay = ['bias', 'LayerNorm.weight']
            p1 = [p for n, p in self.named_parameters() if n == 'ffn_layer.weight']
            p2 = [p for n, p in self.named_parameters() if n == 'ffn_layer.bias']
            p3 = [p for n, p in self.named_parameters() if n.startswith('bert_layer') and any(nd in n for nd in no_decay)]
            p4 = [p for n, p in self.named_parameters() if n.startswith('bert_layer') and not any(nd in n for nd in no_decay)]

            p5 = [p for n, p in self.named_parameters() if n.startswith('bilstm_layer') and any(nd in n for nd in no_decay)]  # possible bilstm after bert
            p6 = [p for n, p in self.named_parameters() if n.startswith('bilstm_layer') and not any(nd in n for nd in no_decay)]  # possible bilstm after bert

            p7 = [p for n, p in self.named_parameters() if n.startswith('interact_layer')]

            self.grouped_params = [
                {'params': p1, 'lr': 1e-3},
                {'params': p2, 'weight_decay': 0.0, 'lr': 1e-3},
                {'params': p3, 'weight_decay': 0.0},
                {'params': p4}
            ]
            if p5 and p6:  # using bilstm after bert
                self.grouped_params += [{'params': p5, 'lr': 1e-3, 'weight_decay': 0.0}]
                self.grouped_params += [{'params': p6, 'lr': 1e-3}]
            if p7:
                self.grouped_params += [{'params': p7, 'lr': 1e-3}]
            self.check_grouped_params(self.grouped_params)
        self.grad_clip = 5.0
        self.total_norm = 0.

    def check_grouped_params(self, grouped_params):
        grouped_params_set = set()
        for d in grouped_params:
            for p in d['params']:
                grouped_params_set.add(id(p))
        assert grouped_params_set == set([id(p) for p in self.parameters()])

    def init_opt(self):
        """Optimizer"""
        if hasattr(self, 'grouped_params'):
            params = self.grouped_params
        else:
            # params = self.parameters()
            params = filter(lambda p: p.requires_grad, self.parameters())
        self.opt = torch.optim.AdamW(params, lr=self.args.lr)  # default weight_decay=1e-2
        # self.opt = AdamW(params, lr=self.lr)  # Transformer impl. default weight_decay=0.

    @property
    def lr_lst(self):
        """Learning Rate List"""
        if not hasattr(self, 'opt'):
            raise ValueError('need model.init_opt() first')
        return [group['lr'] for group in self.opt.param_groups]

    def init_lrs(self, num_step_per_epo=None, epo=None, num_warmup_steps=None):
        """Learing Rate Schedual"""
        if epo is None:
            epo = self.args.num_epochs
        # num_step_per_epo = 209  # onto task0-5
        # num_step_per_epo = 2014  # few task0
        if num_step_per_epo is None:
            num_step_per_epo = 2014
        # num_step_per_epo = (num_training_instancs - 1) // self.args.batch_size + 1
        num_training_steps = num_step_per_epo * epo
        if num_warmup_steps is None:
            ratio = 0.1
            num_warmup_steps = ratio * num_training_steps
        # print(num_training_instancs, epo, ratio, num_step_per_epo, num_training_steps)
        self.lrs = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        # self.lrs = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        # self.lrs = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=num_warmup_steps)

    def context_encoder(self, inputs_dct):
        seq_len = inputs_dct['seq_len']  # seq_len [bat]
        ori_seq_len = inputs_dct['ori_seq_len']

        if self.args.pretrain_mode == 'fine_tuning':
            # with torch.no_grad():  # 不更新bert的话
            bert_outputs = self.bert_layer(input_ids=inputs_dct['input_ids'],
                                           token_type_ids=inputs_dct['bert_token_type_ids'],
                                           attention_mask=inputs_dct['bert_attention_mask'],
                                           output_hidden_states=True,
                                           )


            # ipdb.set_trace()
            # bert_hiddens = bert_outputs.hidden_states[-4:]  # 倒数四层
            # bert_out = sum(bert_hiddens) / len(bert_hiddens)  # average

            bert_out = bert_outputs.last_hidden_state  # 取最后一层

            # 去除bert_output[CLS]和[SEP] # 使用group_aggregating时可以简单只去头尾，漏的[SEP]会自动乘0隐去\
            # seq_len_lst = seq_len.tolist()
            # bert_out_lst = [t for t in bert_out]  # split along batch
            # for i, t in enumerate(bert_out_lst):  # iter along batch
            #     # tensor [len, hid]
            #     bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
            # bert_out = torch.stack(bert_out_lst, 0)  # stack along batch
            bert_out = bert_out[:, 1:-1, :]

            # batch_ori_2_tok = inputs_dct['batch_ori_2_tok']
            # if batch_ori_2_tok.shape[0]:  # 只取子词的第一个字  ENG
            #     bert_out_lst = [t for t in bert_out]
            #     for bdx, t in enumerate(bert_out_lst):
            #         ori_2_tok = batch_ori_2_tok[bdx]
            #         bert_out_lst[bdx] = bert_out_lst[bdx][ori_2_tok]
            #     bert_out = torch.stack(bert_out_lst, 0)

            if inputs_dct['batch_ori_indexes'] is not None:
                ori_indexes = inputs_dct['batch_ori_indexes']
                # print(ori_indexes, ori_indexes.shape, bert_out.shape, sep='\n')
                bert_out = self.group_aggregating(bert_out, ori_indexes, agg_mode='mean')
                # print(bert_out.shape)

            if self.use_bilstm:
                bert_out = self.dropout_layer5(bert_out)  # don't forget
                pack_embed = torch.nn.utils.rnn.pack_padded_sequence(bert_out, ori_seq_len.cpu(), batch_first=True, enforce_sorted=False)
                pack_out, _ = self.bilstm_layer(pack_embed)
                bert_out, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]

        if self.args.pretrain_mode == 'feature_based':
            # if use feature-based finetune
            bert_out = inputs_dct['batch_input_pts']
            bert_out = self.dropout_layer(bert_out)  # don't forget

            pack_embed = torch.nn.utils.rnn.pack_padded_sequence(bert_out, ori_seq_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_out, _ = self.bilstm_layer(pack_embed)
            bert_out, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]

        bert_out = self.dropout_layer2(bert_out)
        return bert_out

    def calc_link_score(self, link_start_hidden, link_end_hidden, fast_impl=True):
        # link_start_hidden [b,l,h]
        # link_end_hidden [b,l,h]
        # return link_dot_prod_scores [b,l-1]
        hidden_size = link_start_hidden.shape[-1]

        if fast_impl:
            # link score 快速计算方式 直接移位相乘再相加(点积)
            link_dot_prod_scores = link_start_hidden[:, :-1, :] * link_end_hidden[:, 1:, :]  # b,l-1,h
            link_dot_prod_scores = torch.sum(link_dot_prod_scores, dim=-1)  # b,l-1
            link_dot_prod_scores = link_dot_prod_scores / hidden_size ** 0.5  # b,l-1
        else:
            # link score 普通计算方式 通过计算矩阵后取对角线 有大量非对角线的无用计算
            link_dot_prod_scores = torch.matmul(link_start_hidden, link_end_hidden.transpose(-1, -2))  # b,l,l
            link_dot_prod_scores = link_dot_prod_scores / hidden_size ** 0.5  # b,e,l,l
            link_dot_prod_scores = torch.diagonal(link_dot_prod_scores, offset=1, dim1=-2, dim2=-1)  # b,l-1

        return link_dot_prod_scores  # b,l-1
        # return torch.relu(link_dot_prod_scores)  # b,l-1

    def calc_loss(self, span_ner_pred_lst, span_ner_tgt_lst, link_valid_span_lst=None, neg_sample_ratio=None):
        if self.loss_type == 'softmax':
            # pred span_ner_pred_lst [*,ent]
            # label span_ner_tgt_lst [*]
            span_loss = self.loss_layer(span_ner_pred_lst, span_ner_tgt_lst)  # 传入[batch,ent] - [batch] target还没变成one-hot
            span_loss = torch.sum(span_loss)

        if self.loss_type == 'sigmoid':
            if neg_sample_ratio:  # e.g. neg_sample=0.2
                neg_mask = (span_ner_tgt_lst.sum(-1) == 0).int()  # [*] 负样本1 正样本0
                rnd = torch.rand(span_ner_tgt_lst.shape[0], device=span_ner_tgt_lst.device) < neg_sample_ratio
                rnd = rnd.int()  # 0.2的概率是1 其余为0
                neg_sample_loss_mask = (1 - neg_mask) + (neg_mask * rnd)
            # pred span_ner_pred_lst [*,ent]
            # label span_ner_tgt_lst [*,ent]
            span_loss = self.loss_layer(span_ner_pred_lst, span_ner_tgt_lst.float())  # [*,ent] [*,ent](target已是onehot) -> [*,ent]
            # span_loss [*,ent]

            if link_valid_span_lst is not None:
                span_loss = span_loss * link_valid_span_lst.unsqueeze(-1)

            span_loss = torch.sum(span_loss, -1)  # [*]
            if neg_sample_ratio:
                span_loss = span_loss * neg_sample_loss_mask.float()
            # span_loss = torch.mean(span_loss)  # 这样loss是0.00x 太小优化不了
            span_loss = torch.sum(span_loss)
        return span_loss

    def calc_f1(self, span_ner_pred_lst, span_ner_tgt_lst):
        if self.loss_type == 'softmax':
            # pred span_ner_pred_lst [*,ent]
            # label span_ner_tgt_lst [*]
            # label span_ner_tgt_lst_onehot [*, ent]
            span_ner_tgt_lst_onehot = torch.nn.functional.one_hot(span_ner_tgt_lst, self.ent_size)  # [*,ent]
            span_ner_pred_lst_onehot = torch.nn.functional.one_hot(torch.argmax(span_ner_pred_lst, dim=-1), self.ent_size)  # [*,ent]

            span_ner_tgt_lst_onehot = span_ner_tgt_lst_onehot[:, 1:]  # 不要O
            span_ner_pred_lst_onehot = span_ner_pred_lst_onehot[:, 1:]  # 不要O

        if self.loss_type == 'sigmoid':
            # pred span_ner_pred_lst [*,ent]
            # label span_ner_tgt_lst [*,ent]
            span_ner_tgt_lst_onehot = span_ner_tgt_lst  # [*,ent]
            span_ner_pred_lst_onehot = (span_ner_pred_lst > 0).int()  # 原本是未经过sigmoid的logit

        # calc f1
        num_gold = torch.sum(span_ner_tgt_lst_onehot)  # [*,ent]
        num_pred = torch.sum(span_ner_pred_lst_onehot)  # [*,ent]
        tp = torch.sum(span_ner_tgt_lst_onehot * span_ner_pred_lst_onehot)
        f1 = torch.tensor(1.) if num_gold == num_pred == 0 else 2 * tp / (num_gold + num_pred + 1e-12)
        return f1.item(), (num_gold.item(), num_pred.item(), tp.item())

    def forward(self, inputs_dct):
        seq_len = inputs_dct['seq_len']  # seq_len [bat]
        ori_seq_len = inputs_dct['ori_seq_len']
        # 如果是sigmoid span_ner_tgt_lst [*, ent] 经过one_hot
        # 如果是softmax span_ner_tgt_lst [*] 直接是类别id id范围为ent + 2 ([PAD] + O)
        span_ner_tgt_lst = inputs_dct['batch_span_tgt']  # span_ner_tgt_lst [len*(len+1)/2]

        # save_info = True
        save_info = False
        if save_info:
            info_dct = {}
            info_dct['exm'] = [{'char_lst': e.char_lst, 'ent_dct': e.ent_dct} for e in inputs_dct['batch_ner_exm']]
            info_dct['ori_seq_len'] = ori_seq_len.detach().cpu().numpy()

        bert_out = self.context_encoder(inputs_dct)
        # =======================Done calc bert output=====================+

        fnn_output_hidden = self.ffn_layer(bert_out)  # [b,l, 2e*h + 2 * h]

        ent_start_hidden, ent_end_hidden = torch.chunk(fnn_output_hidden[:, :, :2 * self.start_size * self.ent_size], 2, dim=-1)
        ent_start_hidden = transpose_for_scores(ent_start_hidden, self.ent_size, self.start_size)  # b,e,l,h
        ent_end_hidden = transpose_for_scores(ent_end_hidden, self.ent_size, self.start_size)  # b,e,l,h

        if self.interact_type == 'self_attn':
            ent_hidden_size = ent_start_hidden.shape[-1]
            span_ner_mat_tensor = torch.matmul(ent_start_hidden, ent_end_hidden.transpose(-1, -2))  # [b,e,l,h] * [b,e,h,l] = [b,e,l,l]
            span_ner_mat_tensor = span_ner_mat_tensor / ent_hidden_size ** 0.5  # b,e,l,l
            span_ner_mat_tensor = span_ner_mat_tensor.permute(0, 2, 3, 1)  # b,l,l,e
        else:
            span_ner_mat_tensor = self.interact_layer(ent_start_hidden, ent_end_hidden)  # [b,l,l,e]

        if save_info: info_dct['span_ner_mat_tensor'] = [e[:l, :l, :] for e, l in zip(span_ner_mat_tensor.detach().cpu().numpy(), info_dct['ori_seq_len'])]

        # batch_link_lst = inputs_dct['batch_link_lst']  # TODO link_lst
        # result_matrix = get_remained_indices_of_spans(inputs_dct['batch_link_lst'], ori_seq_len, only_ret_matrix=True)  # TODO link_lst

        if self.use_slr:
            link_start_hidden, link_end_hidden = torch.chunk(fnn_output_hidden[:, :, 2 * self.start_size * self.ent_size:], 2, dim=-1)
            link_scores = self.calc_link_score(link_start_hidden, link_end_hidden)  # b,l-1
            refined_scores = calc_refined_mat_tensor(link_scores, pooling_type=self.pooling_type, temp=self.args.logsumexp_temp)  # b,l,l,1
            if save_info: info_dct['link_scores'] = [e[:l - 1] for e, l in zip(link_scores.detach().cpu().numpy(), info_dct['ori_seq_len'])]
            if save_info: info_dct['refined_scores'] = [e[:l, :l, :] for e, l in zip(refined_scores.detach().cpu().numpy(), info_dct['ori_seq_len'])]

            if inputs_dct['batch_refine_mask'] is not None:
                refined_scores_mask = inputs_dct['batch_refine_mask'].unsqueeze(-1)
                reversed_refined_scores_mask = 1 - refined_scores_mask
                new_tensor = torch.zeros_like(refined_scores).data.copy_(refined_scores * reversed_refined_scores_mask)
                # print(reversed_refined_scores_mask[0,:15,:15].squeeze())
                # print(refined_scores_mask[0,:15,:15].squeeze())
                # print(new_tensor[0,:15,:15].squeeze(-1))
                new_tensor.require_grad = False
                # import ipdb; ipdb.set_trace()  # 放在某个想调试的函数开始
                refined_scores = refined_scores * refined_scores_mask
                # print(refined_scores[0,:15,:15].squeeze(-1))

                if save_info: info_dct['refined_scores_mask'] = [e[:l, :l, :] for e, l in zip(refined_scores_mask.detach().cpu().numpy(), info_dct['ori_seq_len'])]
                if save_info: info_dct['final_refined_scores'] = [e[:l, :l, :] for e, l in zip(refined_scores.detach().cpu().numpy(), info_dct['ori_seq_len'])]

            # perform refinement
            if self.loss_type == 'softmax':  # minus for 'O' in softmax
                refined_scores = torch.nn.functional.pad(refined_scores, pad=(0, self.ent_size - 1), mode="constant", value=0)  # b,l,l,l  # 只加在O上
                span_ner_mat_tensor = span_ner_mat_tensor - refined_scores

            if self.loss_type == 'sigmoid':  # add for all in sigmoid
                if self.pooling_type.endswith('max'):
                    span_ner_mat_tensor = span_ner_mat_tensor - refined_scores  # logits - max(relu(link))
                    if inputs_dct['batch_refine_mask'] is not None:
                        span_ner_mat_tensor = span_ner_mat_tensor - new_tensor
                elif self.pooling_type.endswith('min'):
                    span_ner_mat_tensor = span_ner_mat_tensor + refined_scores
                    if inputs_dct['batch_refine_mask'] is not None:
                        span_ner_mat_tensor = span_ner_mat_tensor + new_tensor
                else:
                    span_ner_mat_tensor = span_ner_mat_tensor + refined_scores

            if self.loss_type == 'sigmoid':
                norm_link_scores = link_scores
            if self.loss_type == 'softmax':
                norm_link_scores = torch.sigmoid(link_scores)

            if save_info: info_dct['refined_span_ner_mat_tensor'] = [e[:l, :l, :] for e, l in zip(span_ner_mat_tensor.detach().cpu().numpy(), info_dct['ori_seq_len'])]
        else:
            norm_link_scores = None

        # 取出下三角区域 span_ner_pred_lst[*,ent] 通过构造下三角mask(考虑了下三角为0及pad)
        seq_len = ori_seq_len
        len_mask = sequence_mask(seq_len)  # b,l
        matrix_mask = torch.logical_and(torch.unsqueeze(len_mask, 1), torch.unsqueeze(len_mask, 2))  # b,l,l  # 小正方形mask
        score_mat_mask = torch.triu(matrix_mask, diagonal=0)  # b,l,l  # 取出对角线及上三角，其余为0
        # link_valid_span_lst = torch.masked_select(result_matrix, score_mat_mask)  # [*]  TODO link_lst
        # pred_result_matrix = get_remained_indices_of_spans(link_scores >= 0., ori_seq_len, only_ret_matrix=True)  # TODO link_lst
        # pred_link_valid_span_lst = torch.masked_select(pred_result_matrix, score_mat_mask)  # [*]  TODO link_lst
        score_mat_mask = torch.unsqueeze(score_mat_mask, -1).expand(-1, -1, -1, self.ent_size)  # b,l,l,t
        span_ner_pred_lst = torch.masked_select(span_ner_mat_tensor, score_mat_mask)  # 只取True或1组成列表
        span_ner_pred_lst = span_ner_pred_lst.view(-1, self.ent_size)  # [*,ent]

        if save_info:
            info_dct['span_ner_pred_lst'] = span_ner_pred_lst.detach().cpu().numpy()
            info_dct['span_ner_tgt_lst'] = span_ner_tgt_lst.detach().cpu().numpy()
            torch.save(info_dct, os.path.dirname(self.args.exist_ckpt) + '/ptinfo.pt')
            exit(0)

        span_loss = self.calc_loss(span_ner_pred_lst, span_ner_tgt_lst, link_valid_span_lst=None)
        # span_loss = self.calc_loss(span_ner_pred_lst, span_ner_tgt_lst, link_valid_span_lst=link_valid_span_lst)  # TODO link_lst
        # span_ner_pred_lst = span_ner_pred_lst + (-1e8 * (1. - link_valid_span_lst)).unsqueeze(-1)  # TODO link_lst

        # span_loss = self.calc_loss(span_ner_pred_lst, span_ner_tgt_lst, link_valid_span_lst=pred_link_valid_span_lst)  # TODO link_lst
        # span_ner_pred_lst = span_ner_pred_lst + (-1e8 * (1. - pred_link_valid_span_lst)).unsqueeze(-1)  # TODO link_lst
        f1, f1_detail = self.calc_f1(span_ner_pred_lst, span_ner_tgt_lst)

        if self.loss_type == 'softmax':
            span_ner_pred_prob_lst = torch.softmax(span_ner_pred_lst, -1)
        if self.loss_type == 'sigmoid':
            span_ner_pred_prob_lst = torch.sigmoid(span_ner_pred_lst)

        # link_loss = self.link_loss_layer(link_scores, batch_link_lst.float())  # b,l-1  # TODO link_lst
        # link_loss = link_loss * len_mask[:,:-1]
        # self.link_loss = link_loss.mean() * 1000

        return span_loss, span_ner_mat_tensor, span_ner_pred_prob_lst, norm_link_scores, f1, f1_detail

    def predict(self, inputs_dct):
        ori_seq_len = inputs_dct['ori_seq_len']

        bert_out = self.context_encoder(inputs_dct)
        seq_len = ori_seq_len
        # =======================Done calc bert output=====================+

        fnn_output_hidden = self.ffn_layer(bert_out)  # [b,l, 2e*h + 2 * h]
        ent_start_hidden, ent_end_hidden = torch.chunk(fnn_output_hidden[:, :, :2 * self.start_size * self.ent_size], 2, dim=-1)
        ent_start_hidden = transpose_for_scores(ent_start_hidden, self.ent_size, self.start_size)  # b,e,l,h
        ent_end_hidden = transpose_for_scores(ent_end_hidden, self.ent_size, self.start_size)  # b,e,l,h

        # torch.cuda.synchronize()
        # self.args.profiler[2].start()
        cudar1.reset()
        time1.reset()
        if self.interact_type == 'self_attn':
            ent_hidden_size = ent_start_hidden.shape[-1]
            span_ner_mat_tensor = torch.matmul(ent_start_hidden, ent_end_hidden.transpose(-1, -2))  # [b,e,l,h] * [b,e,h,l] = [b,e,l,l]
            span_ner_mat_tensor = span_ner_mat_tensor / ent_hidden_size ** 0.5  # b,e,l,l
            span_ner_mat_tensor = span_ner_mat_tensor.permute(0, 2, 3, 1)  # b,l,l,e
        else:
            span_ner_mat_tensor = self.interact_layer(ent_start_hidden, ent_end_hidden)  # [b,l,l,e]
        # torch.cuda.synchronize()
        # self.args.profiler[2].end()
        time1.record('\nnormal_predict interact')
        cudar1.record('normal_predict interact')
        self.args.m_inter = cudar1.target_value

        if self.use_slr:
            link_start_hidden, link_end_hidden = torch.chunk(fnn_output_hidden[:, :, 2 * self.start_size * self.ent_size:], 2, dim=-1)
            link_scores = self.calc_link_score(link_start_hidden, link_end_hidden)  # b,l-1
            cudar1.reset()
            time1.reset()
            refined_scores = calc_refined_mat_tensor(link_scores, pooling_type=self.pooling_type)  # b,l,l,1
            time1.record('normal_predict calc refined score')
            cudar1.record('normal_predict calc refined score')

            if inputs_dct['batch_refine_mask'] is not None:
                refined_scores_mask = inputs_dct['batch_refine_mask'].unsqueeze(-1)
                reversed_refined_scores_mask = 1 - refined_scores_mask
                new_tensor = torch.zeros_like(refined_scores).data.copy_(refined_scores * reversed_refined_scores_mask)
                new_tensor.require_grad = False
                refined_scores = refined_scores * refined_scores_mask

            # perform refinement
            if self.loss_type == 'softmax':  # minus for 'O' in softmax
                refined_scores = torch.nn.functional.pad(refined_scores, pad=(0, self.ent_size - 1), mode="constant", value=0)  # b,l,l,l  # 只加在O上
                span_ner_mat_tensor = span_ner_mat_tensor - refined_scores

            if self.loss_type == 'sigmoid':  # add for all in sigmoid
                if self.pooling_type.endswith('max'):
                    span_ner_mat_tensor = span_ner_mat_tensor - refined_scores  # logits - max(relu(link))
                    if inputs_dct['batch_refine_mask'] is not None:
                        span_ner_mat_tensor = span_ner_mat_tensor - new_tensor
                elif self.pooling_type.endswith('min'):
                    span_ner_mat_tensor = span_ner_mat_tensor + refined_scores
                    if inputs_dct['batch_refine_mask'] is not None:
                        span_ner_mat_tensor = span_ner_mat_tensor + new_tensor
                else:
                    span_ner_mat_tensor = span_ner_mat_tensor + refined_scores

            if self.loss_type == 'sigmoid':
                norm_link_scores = link_scores
            if self.loss_type == 'softmax':
                norm_link_scores = torch.sigmoid(link_scores)

        else:
            norm_link_scores = None

        # 构造下三角mask 考虑了pad 考虑了下三角为0
        len_mask = sequence_mask(seq_len)  # b,l
        matrix_mask = torch.logical_and(torch.unsqueeze(len_mask, 1), torch.unsqueeze(len_mask, 2))  # b,l,l  # 小正方形mask
        score_mat_mask = torch.triu(matrix_mask, diagonal=0)  # b,l,l  # 取出对角线及上三角，其余为0
        score_mat_mask = torch.unsqueeze(score_mat_mask, -1).expand(-1, -1, -1, self.ent_size)  # b,l,l,t
        span_ner_pred_lst = torch.masked_select(span_ner_mat_tensor, score_mat_mask)  # 只取True或1组成列表
        span_ner_pred_lst = span_ner_pred_lst.view(-1, len(self.ent2id))  # [*,ent] for sigmoid [*,ent+2] for softmax

        if self.loss_type == 'softmax':
            span_ner_pred_prob_lst = torch.softmax(span_ner_pred_lst, -1)
        if self.loss_type == 'sigmoid':
            span_ner_pred_prob_lst = torch.sigmoid(span_ner_pred_lst)

        return span_ner_mat_tensor, span_ner_pred_prob_lst, norm_link_scores  # 返回的是prob不是logit

    def fast_predict(self, inputs_dct):
        ori_seq_len = inputs_dct['ori_seq_len']

        bert_out = self.context_encoder(inputs_dct)
        seq_len = ori_seq_len
        # =======================Done calc bert output=====================+

        fnn_output_hidden = self.ffn_layer(bert_out)  # [b,l, 2e*h + 2 * h]

        ent_start_hidden, ent_end_hidden = torch.chunk(fnn_output_hidden[:, :, :2 * self.start_size * self.ent_size], 2, dim=-1)
        ent_start_hidden = transpose_for_scores(ent_start_hidden, self.ent_size, self.start_size, transopose=False)  # b,e,l,h
        ent_end_hidden = transpose_for_scores(ent_end_hidden, self.ent_size, self.start_size, transopose=False)  # b,e,l,h

        if self.use_slr:
            link_start_hidden, link_end_hidden = torch.chunk(fnn_output_hidden[:, :, 2 * self.start_size * self.ent_size:], 2, dim=-1)
            link_scores = self.calc_link_score(link_start_hidden, link_end_hidden)  # b,l-1
            link_threshold = -7.
            link_threshold = self.args.link_threshold

            # torch.cuda.synchronize()
            # self.args.profiler[2].start()
            cudar1.reset()
            time1.reset()
            ent_start_hidden, ent_end_hidden, indices = fast_decode_0120(link_threshold, link_scores, seq_len, ent_start_hidden, ent_end_hidden)
            time1.record('fast_decode gather for hid')
            cudar1.record('fast_decode gather for hid')
            self.args.m_gather = cudar1.target_value

            """ calc refined score """
            # 正常全量propagation计算
            refined_scores = None

            # 需要 不然准确率会降低
            cudar1.reset()
            time1.reset()
            refined_scores = calc_refined_mat_tensor(link_scores, pooling_type='softmin')  # b,l,l,1
            # time1.record('fast_decode calc refined score')
            # cudar1.record('fast_decode calc refined score')
            # cudar1.reset()
            # time1.reset()
            refined_scores = refined_scores[list(indices)]  # indices[3,*] gather_nd [*,1]  # gather need refined score
            cudar1.record('fast_decode gather for refined score')
            time1.record('fast_decode gather for refined score')
            # 自己手工计算
            # zero_tensor = torch.tensor(0., device=batch_links.device)
            # total_refined_scores = []  # corresponding span refined score
            # for bdx, s, e in total_indices:
            #     if s == e:
            #         total_refined_scores.append(zero_tensor)
            #     else:
            #         inner_links = batch_links[bdx, s: e]
            #         # refined_scores = torch.min(inner_links)  # min
            #         refined_scores = -torch.logsumexp(-inner_links, dim=-1)  # softmin
            #         total_refined_scores.append(refined_scores)
            # refined_scores = torch.stack(total_refined_scores)[:,None]  # [*,1]
            cudar1.reset()
            time1.reset()
            if self.interact_type == 'self_attn':
                ent_hidden_size = ent_start_hidden.shape[-1]
                res_hid = ent_start_hidden * ent_end_hidden  # [*,e,h]
                res_hid = torch.sum(res_hid, dim=-1)  # [*,e]
                span_ner_pred_lst = res_hid / ent_hidden_size ** 0.5  # [*,e]
            else:
                span_ner_pred_lst = self.interact_layer.fast_forward(ent_start_hidden, ent_end_hidden)  # [b,l,l,e]

            time1.record('fast_decode interact')
            cudar1.record('fast_decode interact')
            self.args.m_fast_inter = cudar1.target_value
            # self.args.m_lst.append(self.args.m_gather / self.args.m_inter)
            # self.args.m_lst1.append(self.args.m_fast_inter / self.args.m_inter)

            if refined_scores is not None:
                span_ner_pred_lst = span_ner_pred_lst + refined_scores
            # self.args.profiler[2].end()
            if self.loss_type == 'sigmoid':
                norm_link_scores = link_scores
            if self.loss_type == 'softmax':
                norm_link_scores = torch.sigmoid(link_scores)
        else:
            norm_link_scores = None

        if self.loss_type == 'softmax':
            span_ner_pred_prob_lst = torch.softmax(span_ner_pred_lst, -1)
        if self.loss_type == 'sigmoid':
            span_ner_pred_prob_lst = torch.sigmoid(span_ner_pred_lst)

        return None, span_ner_pred_prob_lst, indices.T, norm_link_scores  # 返回的是prob不是logit  # total_indices.T  [3,*] -> [*,3]

    def fast_predict0825(self, inputs_dct):
        ori_seq_len = inputs_dct['ori_seq_len']

        bert_out = self.context_encoder(inputs_dct)
        seq_len = ori_seq_len
        # =======================Done calc bert output=====================+

        fnn_output_hidden = self.ffn_layer(bert_out)  # [b,l, 2e*h + 2 * h]

        ent_start_hidden, ent_end_hidden = torch.chunk(fnn_output_hidden[:, :, :2 * self.start_size * self.ent_size], 2, dim=-1)
        ent_start_hidden = transpose_for_scores(ent_start_hidden, self.ent_size, self.start_size, transopose=False)  # b,l,e,h
        ent_end_hidden = transpose_for_scores(ent_end_hidden, self.ent_size, self.start_size, transopose=False)  # b,l,e,h
        ent_hidden_size = ent_start_hidden.shape[-1]
        if self.use_slr:
            link_start_hidden, link_end_hidden = torch.chunk(fnn_output_hidden[:, :, 2 * self.start_size * self.ent_size:], 2, dim=-1)
            link_scores = self.calc_link_score(link_start_hidden, link_end_hidden)  # b,l-1
            link_threshold = -7.
            link_threshold = self.args.link_threshold

            # torch.cuda.synchronize()
            # self.args.profiler[2].start()
            cudar1.reset()
            time1.reset()

            result_matrix, indices, start_indices, end_indices = get_remained_indices_of_spans(link_scores > link_threshold, seq_len)
            out = torch.zeros(indices.shape[1], self.ent_size, ent_start_hidden.shape[-1], device=ent_start_hidden.device)
            fast_dot_prod(ent_start_hidden.contiguous(), ent_end_hidden.contiguous(), indices.T.contiguous(), out)
            out = torch.sum(out, -1)
            out = out / ent_hidden_size ** 0.5
            time1.record('taich')
            cudar1.record('taich')
            self.args.m_gather = cudar1.target_value

            """ calc refined score """
            # 正常全量propagation计算
            refined_scores = None

            # 需要 不然准确率会降低
            cudar1.reset()
            time1.reset()
            refined_scores = calc_refined_mat_tensor(link_scores, pooling_type='softmin')  # b,l,l,1
            # time1.record('fast_decode calc refined score')
            # cudar1.record('fast_decode calc refined score')
            # cudar1.reset()
            # time1.reset()
            refined_scores = refined_scores[list(indices)]  # indices[3,*] gather_nd [*,1]  # gather need refined score
            cudar1.record('fast_decode gather for refined score')
            time1.record('fast_decode gather for refined score')
            # 自己手工计算
            # zero_tensor = torch.tensor(0., device=batch_links.device)
            # total_refined_scores = []  # corresponding span refined score
            # for bdx, s, e in total_indices:
            #     if s == e:
            #         total_refined_scores.append(zero_tensor)
            #     else:
            #         inner_links = batch_links[bdx, s: e]
            #         # refined_scores = torch.min(inner_links)  # min
            #         refined_scores = -torch.logsumexp(-inner_links, dim=-1)  # softmin
            #         total_refined_scores.append(refined_scores)
            # refined_scores = torch.stack(total_refined_scores)[:,None]  # [*,1]
            cudar1.reset()
            time1.reset()
            span_ner_pred_lst = out
            time1.record('fast_decode interact')
            cudar1.record('fast_decode interact')
            self.args.m_fast_inter = cudar1.target_value
            # self.args.m_lst.append(self.args.m_gather / self.args.m_inter)
            # self.args.m_lst1.append(self.args.m_fast_inter / self.args.m_inter)

            if refined_scores is not None:
                span_ner_pred_lst = span_ner_pred_lst + refined_scores
            # self.args.profiler[2].end()
            if self.loss_type == 'sigmoid':
                norm_link_scores = link_scores
            if self.loss_type == 'softmax':
                norm_link_scores = torch.sigmoid(link_scores)
        else:
            norm_link_scores = None

        if self.loss_type == 'softmax':
            span_ner_pred_prob_lst = torch.softmax(span_ner_pred_lst, -1)
        if self.loss_type == 'sigmoid':
            span_ner_pred_prob_lst = torch.sigmoid(span_ner_pred_lst)

        return None, span_ner_pred_prob_lst, indices.T, norm_link_scores  # 返回的是prob不是logit  # total_indices.T  [3,*] -> [*,3]


class Bert_Seq(nn.Module):
    def __init__(self, args):
        super(Bert_Seq, self).__init__()
        self.char2id = args.datareader.char2id
        self.tag2id = args.datareader.tag2id
        self.id2char = args.datareader.id2char
        self.id2tag = args.datareader.id2tag
        self.args = args

        self.bert_conf = AutoConfig.from_pretrained(args.bert_model_dir)
        # self.bert_conf = BertConfig.from_pretrained(args.bert_model_dir)

        if args.pretrain_mode == 'fine_tuning':
            # bert_model_dir = 'hfl/chinese-bert-wwm-ext'
            # bert_model_dir = 'bert-base-chinese'
            if args.bert_model_dir.endswith('roberta-base'):
                self.bert_layer = AutoModel.from_pretrained(args.bert_model_dir,
                                                            hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2
                                                            )
            else:
                self.bert_layer = AutoModel.from_pretrained(args.bert_model_dir)
            self.group_aggregating = SequenceGroupAggregating(mode='mean')

        if args.pretrain_mode == 'feature_based':
            self.bilstm_layer = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0
            )

        self.dropout_rate = args.dropout_rate

        self.vocab_size = len(self.char2id)
        self.tag_size = len(self.tag2id)
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.hidden2tag_layer = nn.Linear(self.bert_conf.hidden_size, self.tag_size)
        self.crf_layer = CRF(self.tag_size, batch_first=True)
        self.use_partial_crf = False
        if self.use_partial_crf:
            from crfs import PartialCRF
            self.partial_crf_layer = PartialCRF(num_tags=self.tag_size)

        # print(*[n for n, p in self.named_parameters()], sep='\n')
        count_params(self)

        if self.args.pretrain_mode == 'fine_tuning':
            no_decay = ['bias', 'LayerNorm.weight']
            p1 = [p for n, p in self.named_parameters() if n == 'hidden2tag_layer.weight' or n.startswith('crf_layer')]  # crf
            p2 = [p for n, p in self.named_parameters() if n == 'hidden2tag_layer.bias']  # crf
            p3 = [p for n, p in self.named_parameters() if 'hidden2tag_layer' not in n and not n.startswith('crf_layer') and any(nd in n for nd in no_decay)]
            p4 = [p for n, p in self.named_parameters() if 'hidden2tag_layer' not in n and not n.startswith('crf_layer') and not any(nd in n for nd in no_decay)]

            self.grouped_params = [
                {'params': p1, 'lr': 1e-3},  # crf
                {'params': p2, 'weight_decay': 0.0, 'lr': 1e-3},  # crf
                {'params': p3, 'weight_decay': 0.0},
                {'params': p4},
            ]
            self.check_grouped_params(self.grouped_params)
        self.grad_clip = 5.0
        self.total_norm = 0.

    def check_grouped_params(self, grouped_params):
        grouped_params_set = set()
        for d in grouped_params:
            for p in d['params']:
                grouped_params_set.add(id(p))
        assert grouped_params_set == set([id(p) for p in self.parameters()])

    def init_opt(self):
        """Optimizer"""
        if hasattr(self, 'grouped_params'):
            params = self.grouped_params
        else:
            # params = self.parameters()
            params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.opt = torch.optim.AdamW(params, lr=self.args.lr)  # default weight_decay=1e-2
        # self.opt = AdamW(params, lr=self.lr)  # Transformer impl. default weight_decay=0.

    @property
    def lr_lst(self):
        """Learning Rate List"""
        if not hasattr(self, 'opt'):
            raise ValueError('need model.init_opt() first')
        return [group['lr'] for group in self.opt.param_groups]

    def init_lrs(self, num_step_per_epo=None, epo=None, num_warmup_steps=None):
        if epo is None:
            epo = self.args.num_epochs
        # num_step_per_epo = 209  # onto task0-5
        # num_step_per_epo = 2014  # few task0
        if num_step_per_epo is None:
            num_step_per_epo = 2014
        # num_step_per_epo = (num_training_instancs - 1) // self.args.batch_size + 1
        num_training_steps = num_step_per_epo * epo
        if num_warmup_steps is None:
            ratio = 0.1
            num_warmup_steps = ratio * num_training_steps
        # print(num_training_instancs, epo, ratio, num_step_per_epo, num_training_steps)
        self.lrs = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        # self.lrs = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=num_warmup_steps)

    def context_encoder(self, inputs_dct):
        seq_len = inputs_dct['seq_len']  # seq_len [bat]
        ori_seq_len = inputs_dct['ori_seq_len']

        if self.args.pretrain_mode == 'fine_tuning':
            # with torch.no_grad():  # 不更新bert的话
            bert_outputs = self.bert_layer(input_ids=inputs_dct['input_ids'],
                                           token_type_ids=inputs_dct['bert_token_type_ids'],
                                           attention_mask=inputs_dct['bert_attention_mask'],
                                           output_hidden_states=True,
                                           )

            seq_len_lst = seq_len.tolist()
            bert_out = bert_outputs.last_hidden_state
            # remove [CLS] and [SEP] in bert_output
            bert_out = bert_out[:, 1:-1, :]  # 使用group_aggregating时可以简单只去头尾，漏的[SEP]会自动乘0隐去

            if inputs_dct['batch_ori_indexes'] is not None:
                ori_indexes = inputs_dct['batch_ori_indexes']
                # print(ori_indexes, ori_indexes.shape, bert_out.shape, sep='\n')
                bert_out = self.group_aggregating(bert_out, ori_indexes, agg_mode='mean')
                # print(bert_out.shape)

            bert_out = self.dropout_layer(bert_out)  # don't forget

        if self.args.pretrain_mode == 'feature_based':
            # if use feature-based finetune
            bert_out = inputs_dct['batch_input_pts']
            bert_out = self.dropout_layer(bert_out)  # don't forget

            pack_embed = torch.nn.utils.rnn.pack_padded_sequence(bert_out, ori_seq_len.cpu(), batch_first=True, enforce_sorted=False)  # 补pad
            pack_out, _ = self.bilstm_layer(pack_embed)
            bert_out, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]

        return bert_out

    def forward(self, inputs_dct, train=True):
        tags_ids = inputs_dct['batch_tag_ids']
        bert_out = self.context_encoder(inputs_dct)
        seq_len = inputs_dct['ori_seq_len']

        emission = self.hidden2tag_layer(bert_out)  # [bat,len,tag]
        mask = sequence_mask(seq_len, dtype=torch.uint8)
        if self.use_partial_crf:
            partial_tags_ids = torch.masked_fill(tags_ids, tags_ids == self.tag2id['O'], -1)  # -1是partial_crf中预设的未知标签
            if train:
                crf_loss = self.partial_crf_layer(emission, partial_tags_ids, mask)
            else:
                crf_loss = None
            decode_ids = self.partial_crf_layer.viterbi_decode(emission, mask)  # bsz of tag_list

        else:
            if train:
                crf_log_likelihood = self.crf_layer(emission, tags_ids, mask)
                crf_loss = -crf_log_likelihood
            else:
                crf_loss = None
            decode_ids = self.crf_layer.decode(emission, mask)

        return crf_loss, emission, decode_ids

    def predict(self, inputs_dct):
        bert_out = self.context_encoder(inputs_dct)
        seq_len = inputs_dct['ori_seq_len']

        emission = self.hidden2tag_layer(bert_out)  # [bat,len,tag]
        mask = sequence_mask(seq_len, dtype=torch.uint8)
        if self.use_partial_crf:
            decode_ids = self.partial_crf_layer.viterbi_decode(emission, mask)  # bsz of tag_list
        else:
            decode_ids = self.crf_layer.decode(emission, mask)

        return emission, decode_ids


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    """ mask 句子非pad部分为 1"""
    # lengths [bsz]
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device)  # [len]
    matrix = torch.unsqueeze(lengths, dim=-1)  # [bsz,1]
    mask = row_vector < matrix  # [bsz,len]  auto: [1,len] < [bsz,1]
    mask.type(dtype)  # 类型转换
    return mask  # [bsz,len]


class SequenceGroupAggregating(torch.nn.Module):
    """Aggregating values over steps by groups.

    Parameters
    ----------
    x : torch.FloatTensor (batch, ori_step, hidden)
        The tensor to be aggregate.
    group_by : torch.LongTensor (batch, ori_step)
        The tensor indicating the positions after aggregation.
        Positions being negative values are NOT used in aggregation.
    agg_mode: str
        'mean', 'max', 'min', 'first', 'last'
    agg_step: int
    """

    def __init__(self, mode: str = 'mean'):
        super().__init__()
        if mode.lower() not in ('mean', 'max', 'min', 'first', 'last'):
            raise ValueError(f"Invalid aggregating mode {mode}")
        self.mode = mode

    def forward(self, x: torch.FloatTensor, group_by: torch.LongTensor, agg_step: int = None, agg_mode: str = None):
        if agg_mode is None:
            agg_mode = self.mode
        return SequenceGroupAggregating.sequence_group_aggregating(x, group_by, agg_mode=agg_mode, agg_step=agg_step)

    def extra_repr(self):
        return f"mode={self.mode}"

    @staticmethod
    def sequence_group_aggregating(x: torch.FloatTensor, group_by: torch.LongTensor, agg_mode: str = 'mean', agg_step: int = None):
        """Aggregating values over steps by groups.

        Parameters
        ----------
        x : torch.FloatTensor (batch, ori_step, hidden)
            The tensor to be aggregate.
        group_by : torch.LongTensor (batch, ori_step)  e.g. [[0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 1, 2, -1, -1, -1...-1]]
            The tensor indicating the positions after aggregation.
            The values of `x` with corresponding `group_by` being NEGATIVE are NOT used in aggregation.
            The after-aggregation positions NOT covered by `group_by` are ZEROS.

        agg_mode: str
            'mean', 'max', 'min', 'first', 'last'
        agg_step: int
        """
        if agg_mode.lower() not in ('mean', 'max', 'min', 'first', 'last'):
            raise ValueError(f"Invalid aggregation mode {agg_mode}")

        agg_step = (group_by.max().item() + 1) if agg_step is None else agg_step  # obtain the max valid step (before tokenizing) inner the batch, e.g. 9

        # pos_proj: (agg_step, ori_step)  agg_step: truly word step (9) | ori_step: tokenized subword step (11)
        pos_proj = torch.arange(agg_step, device=group_by.device).unsqueeze(1).expand(-1, group_by.size(1))  # [0,0,..][1,1,..][2,2,..]

        # pos_proj: (batch, agg_step, ori_step)
        pos_proj = (pos_proj.unsqueeze(0) == group_by.unsqueeze(1))  # [T,F,F,F,F,F,F,F,F,F,F],[F,T,T,T,F,F,F,F,F,F,F]

        if agg_mode.lower() in ('mean', 'first', 'last'):
            pos_proj_weight = SequenceGroupAggregating._make_pos_proj_weight(pos_proj, agg_mode=agg_mode)

            # agg_tensor: (batch, agg_step, hidden)
            return pos_proj_weight.bmm(x)  # [batch,agg_step,ori_step] * [batch,ori_step,hidden] = [batch,agg_step,hidden]

        else:
            return SequenceGroupAggregating._execute_pos_proj(x, pos_proj, agg_mode=agg_mode)

    @staticmethod
    def _make_pos_proj_weight(pos_proj: torch.BoolTensor, agg_mode='mean'):
        if agg_mode.lower() == 'mean':
            return torch.nn.functional.normalize(pos_proj.float(), p=1, dim=2)  # [T,F,F,F,F,F,F,F,F,F,F],[F,T,T,T,F,F,F,F,F,F,F] -> [1,0,...],[0,1/3,1/3,1/3,0,...]
        elif agg_mode.lower() == 'first':
            pos_proj_weight = pos_proj & (pos_proj.cumsum(dim=-1) == 1)
            return pos_proj_weight.float()
        elif agg_mode.lower() == 'last':
            pos_proj_weight = pos_proj & (pos_proj.cumsum(dim=-1) == pos_proj.sum(dim=-1, keepdim=True))
            return pos_proj_weight.float()

    @staticmethod
    def _execute_pos_proj(x: torch.FloatTensor, pos_proj: torch.BoolTensor, agg_mode='max'):
        proj_values = []
        for k in range(pos_proj.size(0)):
            curr_proj_values = []
            for curr_pos_proj in pos_proj[k]:
                if curr_pos_proj.sum() == 0:
                    # Set non-covered positions as zeros
                    curr_proj_values.append(torch.zeros(x.size(-1)))
                elif agg_mode.lower() == 'max':
                    curr_proj_values.append(x[k, curr_pos_proj].max(dim=0).values)
                elif agg_mode.lower() == 'min':
                    curr_proj_values.append(x[k, curr_pos_proj].min(dim=0).values)
            proj_values.append(torch.stack(curr_proj_values))
        return torch.stack(proj_values)


class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.type_as(emissions).sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


def aggregate_mask_by_cum(tensor1, mean=True):
    # tensor [batch,len]
    # e.g. [[1,2,3]]
    batch_size, length = tensor1.shape

    # diag_mask = torch.diag_embed(torch.ones([length]), offset=0)  # [l,l]
    # diag_mask = diag_mask[None, ..., None]  # [1,l,l,1]
    # torch.diag_embed(tensor1, )

    diag_t = torch.diag_embed(tensor1, offset=0)  # [b,l,l]
    """diag_t
    [1., 0., 0.]
    [0., 2., 0.]
    [0., 0., 3.]
    """

    cum_t = torch.cumsum(diag_t, dim=-1)  # [b,l,l]
    """cum_t
    [1., 1., 1.]
    [0., 2., 2.]
    [0., 0., 3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2.]
    [1., 1., 1.]
    """

    cum_t = torch.cumsum(cum_t, dim=-2)  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2+3.]
    [1., 1+2., 1+2+3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [1., 1+2., 1+2+3.]
    [0., 2., 2+3.]
    [0., 0., 3.]
    """
    sum_t = cum_t

    """构造相关mask矩阵"""
    ones_matrix = torch.ones(length, length).to(tensor1.device)
    triu_mask = torch.triu(ones_matrix, 0)[None, ...]  # 1,l,l  # 上三角包括对角线为1 其余为0
    ignore_mask = 1. - triu_mask

    if mean:
        # 求平均逻辑
        # 分母： 要除以来求平均
        # e.g. length=3
        heng = torch.arange(1, length + 1).to(tensor1.device)  # [1,2,3]
        heng = heng.unsqueeze(0).repeat((batch_size, 1))  # b,l
        heng = heng.unsqueeze(1).repeat((1, length, 1))  # b,l,l
        """
        [1,2,3]
        [1,2,3]
        [1,2,3]
        """
        shu = torch.arange(0, length).to(tensor1.device)  # [0,1,2]
        shu = shu.unsqueeze(0).repeat((batch_size, 1))  # b,l
        shu = shu.unsqueeze(1).repeat((1, length, 1))  # b,l,l
        shu = shu.transpose(1, 2)
        shu = - shu
        """
        [-0,-0,-0]
        [-1,-1,-1]
        [-2,-2,-2]
        """
        count = heng + shu  # 这里一开始竟然用了- --得正 日
        """
        [1,2,3]
        [0,1,2]
        [-1,0,1]  # 下三角会被mask掉不用管  Note:但是除以不能为0！
        """

        # 把下三角强制变为1 避免计算溢常 因为后面会mask 没关系
        count = count * triu_mask + ignore_mask

        sum_t = sum_t / count

    # 再把下三角强制变为0
    sum_t = sum_t * triu_mask
    return sum_t


def aggregate_mask_by_reduce(tensor1, mode='max', use_soft=True, temp=1):
    """目前在用"""
    # tensor [batch,len]
    # e.g. [[1,2,3]]
    batch_size, length = tensor1.shape

    diag_t = torch.diag_embed(tensor1, offset=0)  # [b,l,l]
    """diag_t
    [1., 0., 0.]
    [0., 2., 0.]
    [0., 0., 3.]
    """

    cum_t = torch.cumsum(diag_t, dim=-1)  # [b,l,l]
    """cum_t
    [1., 1., 1.]
    [0., 2., 2.]
    [0., 0., 3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2.]
    [1., 1., 1.]
    """

    if mode in ['max', 'min']:
        triu_mask = torch.triu(torch.ones([length, length]), diagonal=1).to(tensor1.device)[None, ...]  # 1,l,l
        """triu_mask
        [0., 1., 1.]
        [0., 0., 1.]
        [0., 0., 0.]
        """
        inv_triu_mask = torch.flip(triu_mask, dims=[-1])
        """inv_triu_mask
        [1., 1., 0.]
        [1., 0., 0.]
        [0., 0., 0.]
        """
        if mode == 'max':
            inv_triu_mask = inv_triu_mask * -1e12
            cum_t = cum_t + inv_triu_mask
            """cum_t
            [-inf., -inf., 3.]
            [-inf., 2., 2.]
            [1., 1., 1.]
            """

            if use_soft:
                cum_t = torch.logcumsumexp(cum_t / temp, dim=-2) * temp  # [b,l,l]
            else:
                cum_t, _ = torch.cummax(cum_t, dim=-2)  # [b,l,l]
            """cum_t 0: denote -inf
            [max0., max0., max3.]
            [max0+0., max2+0., max2+3.]
            [max1+0+0., max1+2+0., max1+2+3.]
            """

            cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
            """cum_t
            [max1., max1+2., max1+2+3.]
            [max0., max2., max2+3.]
            [max0., max0., max3.]
            """

            cum_t = torch.triu(cum_t, diagonal=0)  # [b,l,l]
            """cum_t
            [max1., max1+2., max1+2+3.]
            [0., max2., max2+3.]
            [0., 0., max3.]
            """


        elif mode == 'min':
            inv_triu_mask = inv_triu_mask * 1e12
            cum_t = cum_t + inv_triu_mask
            """cum_t
            [inf., inf., 3.]
            [inf., 2., 2.]
            [1., 1., 1.]
            """

            if use_soft:
                cum_t = torch.logcumsumexp(-cum_t / temp, dim=-2) * temp  # [b,l,l]
                cum_t = - cum_t
            else:
                cum_t, _ = torch.cummin(cum_t, dim=-2)  # [b,l,l]
            """cum_t 0: denote inf
            [min0., min0., min3.]
            [min0+0., min2+0., min2+3.]
            [min1+0+0., min1+2+0., min1+2+3.]
            """

            cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
            """cum_t
            [min1., min1+2., min1+2+3.]
            [min0., min2., min2+3.]
            [min0., min0., min3.]
            """

            cum_t = torch.triu(cum_t, diagonal=0)  # [b,l,l]
            """cum_t
            [min1., min1+2., min1+2+3.]
            [0., min2., min2+3.]
            [0., 0., min3.]
            """

    return cum_t


def get_remained_indices_of_spans(links, seq_len, flag=True, only_ret_matrix=False):
    # 取出对应link中的实体的地址。
    # links [b,l]
    bsz, l = links.shape
    cur_device = links.device
    a = torch.tril(torch.ones([bsz, l, l], device=cur_device), diagonal=-1)
    """
    [0., 0., 0.],
    [1., 0., 0.],
    [1., 1., 0.]]
    """

    len_mask = sequence_mask(seq_len - 1)  # b,l
    links = links * len_mask
    b = torch.ones([bsz, l, l], device=cur_device) * links[:, None, :]
    """
    links: [0,1,1]
    [0., 1., 1.],
    [0., 1., 1.],
    [0., 1., 1.]]
    """

    c = torch.triu(b)
    """
    [0., 1., 1.],
    [0., 1., 1.],
    [0., 0., 1.]]
    """

    d = a + c
    """
    [0., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.]]
    """

    e = torch.cumprod(d, dim=-1)
    """
    [0., 0., 0.],
    [1., 1., 1.],
    [1., 1., 1.]]
    """
    e = torch.nn.functional.pad(e, pad=(1, 0, 0, 1), mode="constant", value=1.)  # b,l,l  # 左边和下边增加1

    result_matrix = torch.triu(e)
    """
    [0., 0., 0.],
    [0., 1., 1.],
    [0., 0., 1.]]
    """

    # 注意最后还要过滤掉对角线里面的其实是句子pad的部分.
    result_matrix = result_matrix * sequence_mask(seq_len)[:, None, :]

    if only_ret_matrix:
        return result_matrix

    indices = torch.where(result_matrix == 1.)  # [3,*]
    if not flag:
        indices = torch.stack(indices)  # [3,*]
        return result_matrix, indices

    start_indices = torch.stack([indices[0], indices[1]])  # [2,*]
    # end_indices = torch.stack([indices[0],indices[2]+1])  # 原来没有加pad的话要加1
    end_indices = torch.stack([indices[0], indices[2]])  # [2,*]
    indices = torch.stack(indices)  # [3,*]
    # indices.T -> [*,3]  bsz, start, end
    # print('\ndebug\n')
    # print(links)
    # print(f)
    # print(f.shape)
    # print(start_indices)
    # print(end_indices)
    return result_matrix, indices, start_indices, end_indices


def fast_dot_prod(*_):
    pass


# @ti.kernel
# def fast_dot_prod(
#         batch_start_hid: ti.types.ndarray(field_dim=4),
#         # batch_start_hid: ti.types.Vector.field(50, dtype=ti.f32, shape=(2,10,18)),
#         batch_end_hid: ti.types.ndarray(field_dim=4),
#         indices: ti.types.ndarray(field_dim=2),
#         out: ti.types.ndarray(field_dim=3),
#
# ):
#     num_h = batch_start_hid.shape[-1]
#     num_ents = batch_start_hid.shape[-2]
#     num_spans = indices.shape[0]
#     for nspan, nent, nh in ti.ndrange(num_spans, num_ents, num_h):
#         b = indices[nspan, 0]
#         s = indices[nspan, 1]
#         e = indices[nspan, 2]
#         # for j in range(batch_start_hid.shape[-1]):
#         #     sum_ += batch_start_hid[b, s, nent, j] * batch_end_hid[b, e, nent, j]
#         # start_hid = batch_start_hid[b,s,nent]
#         # end_hid = batch_end_hid[b,e,nent]
#         # v = ti.math.dot(start_hid, end_hid)
#
#         out[nspan, nent, nh] = batch_start_hid[b, s, nent, nh] * batch_end_hid[b, e, nent, nh]


if __name__ == '__main__':
    x = torch.Tensor([-1, -2, -3])
    # x = x[None, ...]
    x = x[None, None, ...].expand(2, 2, -1).reshape(-1, 3)
    print('sum', aggregate_mask_by_cum(x, mean=False))
    print('mean', aggregate_mask_by_cum(x, mean=True))
    print('softmax', aggregate_mask_by_reduce(x, mode='max'))
    print('max', aggregate_mask_by_reduce(x, mode='max', use_soft=False))
    print('softmin', aggregate_mask_by_reduce(x, mode='min'))
    print('min', aggregate_mask_by_reduce(x, mode='min', use_soft=False))

    batch_links = torch.randn([2, 9])
    batch_links = torch.tensor([[.1, .2, .3, .4, .5, .6, .7, .8, .9],
                                [.1, .9, .9, .9, .5, .6, .7, .9, .9]])
    # batch_links = torch.tensor([[0, 1, 1, 0, 0, 0, 1, 0, 0]])
    print(batch_links > 0.8)
    print(batch_links)
    batch_lens = torch.tensor([10, 6])
    batch_start_hid = torch.randn([2, 18, 10, 50])  # [b,e,l,h]
    batch_end_hid = torch.randn([2, 18, 10, 50])  # [b,e,l,h]

    linux = True
    if linux:
        batch_links = batch_links.cuda()
        batch_lens = batch_lens.cuda()
        batch_start_hid = batch_start_hid.cuda()
        batch_end_hid = batch_end_hid.cuda()

    result_matrix, indices, start_indices, end_indices = get_remained_indices_of_spans(batch_links > 0.8, batch_lens)
    print(result_matrix)
    print(indices)

    cudar1.reset()
    time1.reset()
    # field = ti.field(ti.i32, shape=(20, 10, 50))
    # field = ti.Vector.field(1, ti.i32, shape=(20, 10, 50))
    # field.fill(100)
    # print(field[2,3])
    # print(field[2,3].shape)
    batch_start_hid1 = batch_start_hid.transpose(1, 2).contiguous()  # [b,l,e,h]
    batch_end_hid1 = batch_end_hid.transpose(1, 2).contiguous()  # [b,l,e,h]
    num_spans = indices.shape[-1]
    num_ents = 18
    out = torch.zeros(num_spans, num_ents, 50, device=batch_start_hid1.device)
    fast_dot_prod(batch_start_hid1, batch_end_hid1, indices.T.contiguous(), out)
    time1.record('taich')
    cudar1.record('taich')
    print('out', out)
    # exit(0)

    # batch_links1 = batch_links[:1, :]
    # batch_lens1 = batch_lens[:1]
    # result_matrix1, indices1, start_indices, end_indices = get_remained_indices_of_spans(batch_links1 > 0.8, batch_lens1)
    # print(result_matrix1)
    # print(indices1)

    # batch_links2 = batch_links[1:, :]
    # batch_lens2 = batch_lens[1:]
    # result_matrix2, indices2, start_indices, end_indices = get_remained_indices_of_spans(batch_links2 > 0.8, batch_lens2)
    # print(result_matrix2)
    # print(indices2)
    # exit(0)

    fast_decode_0120(0.8, batch_links, batch_lens, batch_start_hid, batch_end_hid)
    exit(0)

    f, indices, start_indices, end_indices = get_remained_indices_of_spans(batch_links > 0.8, batch_lens)
    print(start_indices)
    print(end_indices)
    res = batch_start_hid1[list(start_indices)]
    print(res.shape)
    # batch_start_hid[0].index_select(0, start_indices[1])
    exit(0)

    print('===')
    cudar2.reset()
    time2.reset()
    result_matrix, indices = get_remained_indices_of_spans(batch_links > 0.8, batch_lens, flag=False)
    print(result_matrix.shape)
    length = batch_start_hid.shape[2]
    batch_start_hid1 = batch_start_hid.transpose(1, 2).unsqueeze(2).expand(-1, -1, length, -1, -1)
    print(batch_start_hid1.shape)
    batch_start_hid1 = torch.masked_select(batch_start_hid1, result_matrix[..., None, None].bool())
    print(batch_start_hid1.shape)
    batch_start_hid1.view(indices[0].shape[0], 18, 50)
    time2.record('fast_decode gather_by_indices')
    cudar2.record('fast_decode gather_by_indices')
    exit(0)
