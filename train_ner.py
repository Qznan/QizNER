# !/usr/bin/env python
# coding=utf-8
"""
model: span-level and seq-labeled ner model
started from 2021/1
"""
import os, time, sys, argparse, copy, random, subprocess

sys.path = ['.'] + sys.path
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import ipdb
import logging, pprint
from rich import print
import datautils as utils
from modules import Bert_Span, Bert_Seq
from data_reader import NerDataReader

utils.setup_seed(1111, np, torch)


def fmt(n):
    return round(n * 100, 4)


class Trainer:
    def __init__(self, args, arch='seq', exist_ckpt=None, evaluate=False):
        """
        evaluate True will eval the test dataset or passing args.infer_dataset
        """
        self.args = args
        self.evaluate = evaluate
        self.arch = arch
        self.use_gpu = args.use_gpu
        self.device = args.device
        self.use_tqdm = True

        time_series = utils.get_curr_time_str('%y_%m_%d_%H_%M')
        model_info = f'{args.corpus}_{arch}'

        if arch == 'span':
            model_info += f'_{args.interact_type}_{args.span_loss_type}'
            if args.use_slr:
                model_info += f'_SLR_{args.pooling_type}'
        args.info = '_' + args.info if args.info else ''
        self.curr_ckpt_dir = args.curr_ckpt_dir = Path('') / args.ckpt_dir / f'{time_series}_{model_info}{args.info}'
        # utils.print_vars(args, maxlen=200)
        logger.info(" ".join(sys.argv))
        logger.info(f'args:\n%s', pprint.pformat(args.__dict__))
        # print(args.__dict__)  # rich print
        # print("======================")

        Model = {
            'span': Bert_Span,
            'seq': Bert_Seq,
        }.get(arch, None)

        self.model = Model(args)
        self.model.to(self.device)

        if exist_ckpt is not None:
            args.exist_ckpt = exist_ckpt
            map_location = torch.device('cuda') if self.use_gpu else torch.device('cpu')
            exist_state_dict = torch.load(exist_ckpt, map_location=map_location)
            if getattr(args, 'exist_ckpt_slr_but_curr_nonslr', None):
                exist_state_dict['ffn_layer.weight'] = exist_state_dict['ffn_layer.weight'][:-100, :]
                exist_state_dict['ffn_layer.bias'] = exist_state_dict['ffn_layer.bias'][:-100]
            self.model.load_state_dict(exist_state_dict)
            print(f'load exist model ckpt success. {exist_ckpt} ')

        self.saved_exm_extn_attrs = []
        self.saved_exm_extn_attrs += ['char_lst']
        self.saved_exm_extn_attrs += ['sub_tokens', 'ori_indexes']

    def train(self):
        args = self.args
        self.best_test_epo = 0
        self.best_test_step = 0
        self.best_test_f1 = -1
        self.best_dev_epo = 0
        self.best_dev_step = 0
        self.best_dev_f1 = -1
        self.test_f1_in_best_dev = -1
        self.best_test_detail_info_str = ''
        self.best_dev_detail_info_str = ''

        self.metrics_jsonl = []
        arch = self.arch
        self.train_data_loader, self.test_data_loader, self.dev_data_loader = None, None, None
        if getattr(args, 'train_dataset', None):
            self.train_data_loader = torch.utils.data.DataLoader(args.train_dataset, batch_size=args.batch_size, shuffle=True,
                                                                 collate_fn=args.datareader.get_batcher_fn(gpu=self.use_gpu, arch=arch, device=self.device))
            logger.info(f'train num: {len(args.train_dataset)}')
            self.train_num_steps = (len(args.train_dataset) - 1) // args.batch_size + 1
            self.total_num_steps = args.num_epochs * self.train_num_steps
        if getattr(args, 'dev_dataset', None):
            self.dev_data_loader = torch.utils.data.DataLoader(args.dev_dataset, batch_size=args.batch_size, shuffle=False,
                                                               collate_fn=args.datareader.get_batcher_fn(gpu=self.use_gpu, arch=arch, device=self.device))
            logger.info(f'dev num: {len(args.dev_dataset)}')
        if getattr(args, 'test_dataset', None):
            self.test_data_loader = torch.utils.data.DataLoader(args.test_dataset, batch_size=args.batch_size, shuffle=False,
                                                                collate_fn=args.datareader.get_batcher_fn(gpu=self.use_gpu, arch=arch, device=self.device))
            logger.info(f'test num: {len(args.test_dataset)}')

        if self.evaluate and getattr(args, 'infer_dataset', None):
            self.test_data_loader = torch.utils.data.DataLoader(args.infer_dataset, batch_size=args.batch_size, shuffle=False,
                                                                collate_fn=args.datareader.get_batcher_fn(gpu=self.use_gpu, arch=arch, device=self.device))
            logger.info(f'infer num: {len(args.infer_dataset)}')

        self.curr_step = 0
        # self.opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.model.init_opt()
        self.model.init_lrs(num_step_per_epo=self.train_num_steps, epo=args.num_epochs, num_warmup_steps=args.num_warmup_steps)

        self.run_epo = {
            'span': self.run_epo_span,
            'seq': self.run_epo_seq,
        }.get(self.arch, None)
        self.run()

    def run(self):
        if self.evaluate:
            test_f1, test_ef1, exm_lst, *_ = self.run_epo(1, mode='test')
            utils.NerExample.save_to_jsonl(exm_lst, self.curr_ckpt_dir / 'infer_exm_lst.jsonl', external_attrs=self.saved_exm_extn_attrs)
            print(test_f1, test_ef1)
            return
        for epo in range(1, self.args.num_epochs + 1):
            train_f1, train_ef1, train_exm_lst, train_detail_info_str = self.run_epo(epo, mode='train')
            logger.info(f'train detail info:\n{train_detail_info_str}')
            if self.dev_data_loader is not None and self.test_data_loader is not None:  # 没有验证集只有测试集
                is_better = self.eval_dev(epo)
            elif self.dev_data_loader is None and self.test_data_loader is not None:  # 有验证集有测试集
                is_better = self.eval_test(epo)
            else:
                raise NotImplementedError
            if is_better:
                utils.del_if_exists(self.curr_ckpt_dir, pattern='train_*_exm_lst.jsonl')
                utils.NerExample.save_to_jsonl(train_exm_lst, self.curr_ckpt_dir / f'train_{epo}_{self.curr_step}_exm_lst.jsonl', external_attrs=self.saved_exm_extn_attrs)
            # if self.decide_early_stop():
            #     print('early_stop!')
            #     break

        if self.dev_data_loader is not None: logger.info(f'best dev detail info: {self.best_ckpt_full_path}\n{self.best_dev_detail_info_str}')
        if self.test_data_loader is not None: logger.info(f'best test detail info: {self.best_ckpt_full_path}\n{self.best_test_detail_info_str}')

    @property
    def best_ckpt_full_path(self):
        if self.dev_data_loader is not None:
            return self.curr_ckpt_dir / f'best_dev_model_{self.best_dev_epo}_{self.best_dev_step}.ckpt'
        else:
            return self.curr_ckpt_dir / f'best_test_model_{self.best_test_epo}_{self.best_test_step}.ckpt'

    def eval_dev(self, epo, save_best_test_on_best_dev=True):
        dev_f1, dev_ef1, dev_exm_lst, dev_detail_info_str = self.run_epo(epo, mode='dev')
        logger.info(f'dev detail info:\n{dev_detail_info_str}')
        test_f1, test_ef1, test_exm_lst, test_detail_info_str = self.run_epo(epo, mode='test')
        logger.info(f'test detail info:\n{test_detail_info_str}')
        if save_best_test_on_best_dev:
            is_better = False
            if dev_ef1 > self.best_dev_f1:
                is_better = True
                utils.del_if_exists(self.curr_ckpt_dir, pattern='best_dev_model_*_*.ckpt')  # delete old ckpt
                self.best_dev_epo, self.best_dev_step, self.best_dev_f1, self.test_f1_in_best_dev = epo, self.curr_step, dev_ef1, test_ef1
                torch.save(self.model.state_dict(), self.curr_ckpt_dir / f'best_dev_model_{self.best_dev_epo}_{self.best_dev_step}.ckpt')
                utils.del_if_exists(self.curr_ckpt_dir, pattern='dev_*_exm_lst.jsonl')
                utils.del_if_exists(self.curr_ckpt_dir, pattern='test_*_exm_lst.jsonl')
                utils.NerExample.save_to_jsonl(dev_exm_lst, self.curr_ckpt_dir / f'dev_{epo}_{self.curr_step}_exm_lst.jsonl', external_attrs=self.saved_exm_extn_attrs)
                utils.NerExample.save_to_jsonl(test_exm_lst, self.curr_ckpt_dir / f'test_{epo}_{self.curr_step}_exm_lst.jsonl', external_attrs=self.saved_exm_extn_attrs)
                self.best_dev_detail_info_str = dev_detail_info_str
                self.best_test_detail_info_str = test_detail_info_str
            logger.info(f'curr_step:{self.curr_step} best_dev_f1:{self.best_dev_f1} best_dev_epo:{self.best_dev_epo}_{self.best_dev_step} test_f1_in_best_dev:{self.test_f1_in_best_dev}\n')
            return is_better
        else:  # will save best test ignoring dev, dev is only for shown
            dev_is_better = False
            if dev_ef1 > self.best_dev_f1:
                dev_is_better = True
                utils.del_if_exists(self.curr_ckpt_dir, pattern='best_dev_model_*_*.ckpt')  # delete old ckpt
                self.best_dev_epo, self.best_dev_step, self.best_dev_f1, self.test_f1_in_best_dev = epo, self.curr_step, dev_ef1, test_ef1
                torch.save(self.model.state_dict(), self.curr_ckpt_dir / f'best_dev_model_{self.best_dev_epo}_{self.best_dev_step}.ckpt')
                utils.del_if_exists(self.curr_ckpt_dir, pattern='dev_*_exm_lst.jsonl')
                utils.del_if_exists(self.curr_ckpt_dir, pattern='test_in_dev_*_exm_lst.jsonl')
                utils.NerExample.save_to_jsonl(dev_exm_lst, self.curr_ckpt_dir / f'dev_{epo}_{self.curr_step}_exm_lst.jsonl', external_attrs=self.saved_exm_extn_attrs)
                utils.NerExample.save_to_jsonl(test_exm_lst, self.curr_ckpt_dir / f'test_in_dev_{epo}_{self.curr_step}_exm_lst.jsonl', external_attrs=self.saved_exm_extn_attrs)
                self.best_dev_detail_info_str = dev_detail_info_str
                self.best_test_detail_info_str = test_detail_info_str
            logger.info(f'curr_step:{self.curr_step} best_dev_f1:{self.best_dev_f1} best_dev_epo:{self.best_dev_epo}_{self.best_dev_step} test_f1_in_best_dev:{self.test_f1_in_best_dev}\n')

            test_is_better = False
            if test_ef1 > self.best_test_f1:
                test_is_better = True
                utils.del_if_exists(self.curr_ckpt_dir, pattern='best_test_model_*_*.ckpt')  # delete old ckpt
                self.best_test_epo, self.best_test_step, self.best_test_f1 = epo, self.curr_step, test_ef1
                torch.save(self.model.state_dict(), self.curr_ckpt_dir / f'best_test_model_{self.best_test_epo}_{self.best_test_step}.ckpt')
                utils.del_if_exists(self.curr_ckpt_dir, pattern='test_*_exm_lst.jsonl')
                utils.NerExample.save_to_jsonl(test_exm_lst, self.curr_ckpt_dir / f'test_{epo}_{self.curr_step}_exm_lst.jsonl', external_attrs=self.saved_exm_extn_attrs)
                self.best_test_detail_info_str = test_detail_info_str
            logger.info(f'curr_step:{self.curr_step} best_test_f1:{self.best_test_f1} best_test_epo:{self.best_test_epo}_{self.best_test_step}\n')
            return dev_is_better or test_is_better

    def eval_test(self, epo):
        test_f1, test_ef1, test_exm_lst, test_detail_info_str = self.run_epo(epo, mode='test')
        logger.info(f'test detail info:\n{test_detail_info_str}')
        is_better = False
        if test_ef1 > self.best_test_f1:
            is_better = True
            utils.del_if_exists(self.curr_ckpt_dir, pattern='best_test_model_*_*.ckpt')  # delete old ckpt
            self.best_test_epo, self.best_test_step, self.best_test_f1 = epo, self.curr_step, test_ef1
            torch.save(self.model.state_dict(), self.curr_ckpt_dir / f'best_test_model_{self.best_test_epo}_{self.best_test_step}.ckpt')
            utils.del_if_exists(self.curr_ckpt_dir, pattern='test_*_exm_lst.jsonl')
            utils.NerExample.save_to_jsonl(test_exm_lst, self.curr_ckpt_dir / f'test_{epo}_{self.curr_step}_exm_lst.jsonl', external_attrs=self.saved_exm_extn_attrs)
            self.best_test_detail_info_str = test_detail_info_str
        logger.info(f'curr_step:{self.curr_step} best_test_f1:{self.best_test_f1} best_test_epo:{self.best_test_epo}_{self.best_test_step}\n')
        return is_better

    def after_per_step(self, epo, mode):
        if mode == 'train':
            self.curr_step += 1
            if self.args.eval_every_step != 0 and self.curr_step >= self.args.eval_after_step and self.curr_step % self.args.eval_every_step == 0:
                if not self.use_tqdm: print()
                if self.dev_data_loader is not None and self.test_data_loader is not None:  # 没有验证集只有测试集
                    self.eval_dev(epo)
                elif self.dev_data_loader is None and self.test_data_loader is not None:  # 有验证集有测试集
                    self.eval_test(epo)
                else:
                    raise NotImplementedError
                self.model.train()

    def decide_early_stop(self, patience_iter=5, metric_item='f', datatype='test', greater_is_better=True):
        # 判断是否早停
        metric_item_lst = [e['info'][metric_item] for e in self.metrics_jsonl if e['mode'] == datatype]  # 已经根据步数排号序了
        if not greater_is_better: metric_item_lst = [-e for e in metric_item_lst]
        max_idx = np.argmax(metric_item_lst)
        if len(metric_item_lst) - patience_iter > max_idx:
            return True
        else:
            return False

    def profile(self, mode='train'):
        data_loader = {
            'train': self.train_data_loader,
            'test': self.test_data_loader,
            'dev': self.dev_data_loader,
        }.get(mode, None)
        self.model.train() if mode == 'train' else self.model.eval()

        with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
            num_test_batch = 20
            t0 = time.time()
            dummy_loader = [batch for _, batch in zip(range(num_test_batch), data_loader)]  # prior fetch dataloader
            logger.info(f"Data loading time: {time.time() - t0:.3f}s")
            t0 = time.time()
            for inputs in dummy_loader:
                if self.arch == 'span':
                    span_loss, span_ner_mat_tensor, batch_predict, conj_dot_product_score, f1, f1_detail = self.model(inputs)
                elif self.arch == 'seq':
                    crf_loss, emission, decode_ids = self.model(inputs)
            logger.info(f"Model training time: {time.time() - t0:.3f}s")

        sort_by = "cuda_time_total" if self.use_gpu else "cpu_time_total"
        prof_table = prof.key_averages().table(sort_by=sort_by, row_limit=200)
        logger.info(f"\n{prof_table}")
        ipdb.set_trace()
        return prof

    def run_epo_span(self, epo, mode='train'):
        # logger.info(f'\n{mode}...{"=" * 60}')
        logger.info(utils.header_format(f'{mode} {epo} epo', sep='='))
        args = self.args
        data_loader = {
            'train': self.train_data_loader,
            'test': self.test_data_loader,
            'dev': self.dev_data_loader,
        }.get(mode, None)

        if data_loader is None:
            return None, None

        self.model.train() if mode == 'train' else self.model.eval()

        total_steps = len(data_loader)

        f1_meaner = utils.F1_Meaner()
        loss_meaner = utils.Meaner()

        iterator = tqdm(data_loader, ncols=200, dynamic_ncols=True) if self.use_tqdm else data_loader
        for num_steps, inputs in enumerate(iterator, start=1):
            step_start_time = time.time()
            batch_ner_exm = inputs['batch_ner_exm']
            batch_seq_len = inputs['ori_seq_len']  # TODO

            if mode == 'train':
                span_loss, span_ner_mat_tensor, batch_predict, conj_dot_product_score, f1, f1_detail = self.model(inputs)
            else:
                if args.use_refine_mask: inputs['batch_refine_mask'] = None
                with torch.no_grad():  # 不计算梯度 更进一步节约内存 与train() eval()无关
                    span_loss, span_ner_mat_tensor, batch_predict, conj_dot_product_score, f1, f1_detail = self.model(inputs)
                    # span_loss, span_ner_mat_tensor, batch_span_ner_pred_lst, conj_dot_product_score, f1, f1_detail = self.model.eval_fast_forward(inputs)
            # span_loss_ = span_loss
            # span_loss = span_loss_ + self.model.link_loss
            if mode == 'train':
                self.model.opt.zero_grad()
                span_loss.backward()
                if self.model.grad_clip is None:
                    self.model.total_norm = 0
                else:
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=5.)
                    self.model.total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.grad_clip)
                self.model.opt.step()
                self.model.lrs.step()

            f1_meaner.add(*f1_detail)
            loss_meaner.add(float(span_loss))

            if mode in ['test', 'dev'] or epo > 1:
                # batch中切分每个的span_level_lst
                batch_predict_lst = torch.split(batch_predict.detach().cpu(), (batch_seq_len * (batch_seq_len + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]
                for exm, length, pred_prob in zip(batch_ner_exm, batch_seq_len.tolist(), batch_predict_lst):
                    if args.span_loss_type == 'softmax':
                        negative_set = {0}
                        exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst(pred_prob.numpy(), length, self.args.datareader.id2ent, negative_set=negative_set)  # softmax
                    if args.span_loss_type == 'sigmoid':
                        exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst_sigmoid(pred_prob.numpy(), length, self.args.datareader.id2ent)  # sigmoid
                    # print(exm.pred_ent_dct)
            else:
                for exm in batch_ner_exm:
                    exm.pred_ent_dct = {}

            if self.use_tqdm:
                iterator.set_description(
                    f'Ep{epo} Step{self.curr_step} '
                    # f'step:{num_steps}/{total_steps} '
                    f'CurLoss:{span_loss:.3f} EpoLoss:{loss_meaner.v:.3f} '
                    # f'CurLoss:{span_loss:.3f}|{span_loss_:.3f},{self.model.link_loss:.3f} EpoLoss:{loss_meaner.v:.3f} '
                    f'CurF1:{f1:.3f} '
                    f'Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1:{f1_meaner.f1:.3f} '
                    # f'LR:({"/".join(f"{lr:.6f}" for lr in self.model.lr_lst)}))'
                    f'LR:{self.model.lr_lst[0]:.6f} '
                    f'GNorm: {self.model.total_norm:.3f} Bsz:{data_loader.batch_size}'
                )
            else:
                print(f'\repo:{epo} step:{num_steps}/{total_steps} '
                      f'cur_loss:{span_loss:.3f} epo_loss:{loss_meaner.v:.3f} curr_lr:{self.model.lr_lst[0]:.6f} '
                      f'cur_f1:{f1:.3f} epo_f1:{f1_meaner.v:.3f} '
                      f'sec/step:{time.time() - step_start_time:.2f}',
                      end=f'{os.linesep if num_steps == total_steps else ""}',
                      )

            self.after_per_step(epo, mode)

            # if self.conf['span_layer_type'] == 'self_attn_mask_mean':
            #     # conj_dot_product_score = conj_dot_product_score.cpu().detach().numpy().tolist()
            #     for bdx, exm in enumerate(batch_ner_exm):
            #         exm.conj_scores, conj_res = exm.get_conj_info(conj_dot_product_score[bdx], decimal=6)

            # 输出连接分数
            # print()
            # out = []
            # conj_dot_product_score = conj_dot_product_score.tolist()
            # for bdx, exm in enumerate(batch_ner_exm):modu
            #     conj_res = exm.get_conj_info(conj_dot_product_score[bdx])
            #     out.append(conj_res)
            #     out.append(exm.get_ent_lst(for_human=True))
            #     out.append(exm.get_pred_ent_lst(for_human=True))
            #     out.append(exm.text)
            #     out.append(conj_dot_product_score[bdx])
            # print(*out, sep='\n')
            # input('请输入任意键继续')
            # # exit(0)

        logger.info(
            f'Ep{epo} Step{self.curr_step} '
            f'EpoLoss:{loss_meaner.v:.3f} '
            f'Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1:{f1_meaner.f1:.3f} '
            f'LR:{self.model.lr_lst[0]:.6f} '
            f'GNorm: {self.model.total_norm:.3f} Bsz:{data_loader.batch_size}'
        )

        exm_lst = data_loader.dataset.instances

        metric_info = {'mode': mode, 'epo': epo, 'step': self.curr_step, 'info': {'loss': round(loss_meaner.v, 4)}}

        # self.metrics_jsonl.append(metric_info)
        # self.metrics_jsonl.sort(key=lambda e: e['epo'])
        # self.metrics_jsonl.sort(key=lambda e: e['mode'])
        # utils.save_jsonl(self.metrics_jsonl, (self.curr_ckpt_dir / 'metrics.jsonl'), verbose=False)
        # return 0., -2, [], ''

        ep, er, ef, detail_info_str_e, *_ = utils.NerExample.eval(exm_lst, verbose=False, use_flat_pred_ent_dct=False)
        logger.info(f'Ori metric: {mode} p|r|f: {ep:.3%}|{er:.3%}|{ef:.3%}')
        ep, er, ef = map(fmt, (ep, er, ef))
        metric_info['info'].update({'ep': ep, 'er': er, 'ef': ef, })

        if args.flat:
            p, r, f, detail_info_str, *_ = utils.NerExample.eval(exm_lst, verbose=False, use_flat_pred_ent_dct=True)
            logger.info(f'Flat metric: {mode} p|r|f: {p:.3%}|{r:.3%}|{f:.3%}')
            p, r, f = map(fmt, (p, r, f))
            metric_info['info'].update({'p': p, 'r': r, 'f': f, })

        # if conf['flat'] and self.conf['span_layer_type'] == 'self_attn_mask_mean':
        #     tmp_exm_lst = copy.deepcopy(exm_lst)
        #     for exm in tmp_exm_lst:
        #         exm.pred_ent_dct = exm.get_flat_pred_ent_dct_by_conj_scores(exm.conj_scores)
        #     cp, cr, cf, detail_info_str_c, *_ = utils.NerExample.eval(tmp_exm_lst, verbose=False, use_flat_pred_ent_dct=False)
        #     print(f'ori_metric: {mode} p-r-f: {cp:.3%}-{cr:.3%}-{cf:.3%}')
        #     cp, cr, cf = map(fmt, (cp, cr, cf))
        #     metric_info['info'].update({'cp': cp, 'cr': cr, 'cf': cf, })

        self.metrics_jsonl.append(metric_info)
        self.metrics_jsonl.sort(key=lambda e: e['epo'])
        self.metrics_jsonl.sort(key=lambda e: e['mode'])
        utils.save_jsonl(self.metrics_jsonl, (self.curr_ckpt_dir / 'metrics.jsonl'), verbose=False)
        utils.save_args_to_json_file(args, self.curr_ckpt_dir / 'args.json')
        utils.list2file(self.args.datareader.ent_lst, self.curr_ckpt_dir / 'ent_lst.txt', verbose=False)
        if args.flat:
            return 0., f, exm_lst, detail_info_str
        else:
            return 0., ef, exm_lst, detail_info_str_e

    def run_epo_seq(self, epo, mode='train'):
        logger.info(utils.header_format(f'{mode} {epo} epo', sep='='))
        args = self.args
        data_loader = {
            'train': self.train_data_loader,
            'test': self.test_data_loader,
            'dev': self.dev_data_loader,
        }.get(mode, None)

        if data_loader is None:
            return None, None

        self.model.train() if mode == 'train' else self.model.eval()

        total_steps = len(data_loader)
        loss_meaner = utils.Meaner()

        iterator = tqdm(data_loader, ncols=200, dynamic_ncols=True) if self.use_tqdm else data_loader
        for num_steps, inputs in enumerate(iterator, start=1):
            step_start_time = time.time()
            batch_ner_exm = inputs['batch_ner_exm']

            if mode == 'train':
                crf_loss, emission, decode_ids = self.model(inputs)
                # decode_ids: List
            else:
                with torch.no_grad():  # 不计算梯度 更进一步节约内存 与train() eval()无关
                    crf_loss, emission, decode_ids = self.model(inputs)
            emission_probs = torch.softmax(emission, -1).cpu().detach().numpy()  # [bat,len,tag]

            if mode == 'train':
                self.model.opt.zero_grad()
                crf_loss.backward()
                if self.model.grad_clip is None:
                    self.model.total_norm = 0
                else:
                    self.model.total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.grad_clip)
                self.model.opt.step()
                self.model.lrs.step()

            loss_meaner.add(float(crf_loss))

            if mode in ['test', 'dev'] or epo > 0:
                for exm, decode_ids_, emission_probs_ in zip(batch_ner_exm, decode_ids, emission_probs):
                    tag_lst = [self.args.datareader.id2tag[tag_id] for tag_id in decode_ids_]
                    emission_prob = [emission_probs_[i, did] for i, did in enumerate(decode_ids_)]
                    # assert len(tag_lst) == len(exm.char_lst)
                    pred_ent_dct, _ = utils.NerExample.extract_entity_by_tags(tag_lst)
                    for k, v_lst in pred_ent_dct.items():
                        for e in v_lst:
                            start, end = e[0], e[1]
                            mean_emission_prob = round(float(np.mean(emission_prob[start:end])), 4)
                            e.append(mean_emission_prob)  # 采用直接平均要素中所有token对应输出tag的发射概率
                            # e.append(1.)  # 假设概率为1
                    exm.pred_ent_dct = pred_ent_dct
            else:
                for exm in batch_ner_exm:
                    exm.pred_ent_dct = {}

            if self.use_tqdm:
                iterator.set_description(
                    f'Ep{epo} Step{self.curr_step} '
                    # f'step:{num_steps}/{total_steps} '
                    f'CurLoss:{crf_loss:.3f} EpoLoss:{loss_meaner.v:.3f} '
                    f'LR:{self.model.lr_lst[0]:.6f} '
                    f'GNorm: {self.model.total_norm:.3f} Bsz:{data_loader.batch_size}'
                )
            else:
                print(f'\repo:{epo} step:{num_steps}/{total_steps} '
                      f'cur_loss:{crf_loss:.3f} epo_loss:{loss_meaner.v:.3f} curr_lr:{self.model.lr_lst[0]:.6f} '
                      f'sec/step:{time.time() - step_start_time:.2f}',
                      end=f'{os.linesep if num_steps == total_steps else ""}',
                      )

            self.after_per_step(epo, mode)

        logger.info(
            f'Ep{epo} Step{self.curr_step} '
            f'CurLoss:{crf_loss:.3f} EpoLoss:{loss_meaner.v:.3f} '
            f'LR:{self.model.lr_lst[0]:.6f} '
            f'GNorm: {self.model.total_norm:.3f} Bsz:{data_loader.batch_size}'
        )

        exm_lst = data_loader.dataset.instances

        metric_info = {'mode': mode, 'epo': epo, 'step': self.curr_step, 'info': {'loss': round(loss_meaner.v, 4)}}

        p, r, f, detail_info_str, *_ = utils.NerExample.eval(exm_lst, verbose=False, use_flat_pred_ent_dct=True)
        print(f'{mode} p-r-f: {p:.3%}-{r:.3%}-{f:.3%}')
        logger.info(f'metric: {mode} p|r|f: {p:.3%}|{r:.3%}|{f:.3%}')
        p, r, f = map(fmt, (p, r, f))
        metric_info['info'].update({'p': p, 'r': r, 'f': f})

        self.metrics_jsonl.append(metric_info)
        self.metrics_jsonl.sort(key=lambda e: e['epo'])
        self.metrics_jsonl.sort(key=lambda e: e['mode'])
        utils.save_jsonl(self.metrics_jsonl, (self.curr_ckpt_dir / 'metrics.jsonl'), verbose=False)
        utils.save_args_to_json_file(args, self.curr_ckpt_dir / 'args.json')
        utils.list2file(self.args.datareader.ent_lst, self.curr_ckpt_dir / 'ent_lst.txt', verbose=False)
        return 0, f, exm_lst, detail_info_str

    def predict_span(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                  collate_fn=args.datareader.get_batcher_fn(gpu=self.use_gpu, mode=self.arch, device=self.device))

        print(f'infer num: {len(dataset)}')
        self.infer_num_steps = (len(dataset) - 1) // args.batch_size + 1

        self.run_epo = {
            'span': self.run_epo_span,
            'seq': self.run_epo_seq,
        }.get(self.arch, None)

        self.model.eval()
        iterator = tqdm(data_loader, ncols=200) if self.use_tqdm else data_loader
        for num_steps, inputs in enumerate(iterator, start=1):
            batch_ner_exm = inputs['batch_ner_exm']
            batch_seq_len = inputs['ori_seq_len']

            if args.use_refine_mask: inputs['batch_refine_mask'] = None
            with torch.no_grad():  # 不计算梯度 更进一步节约内存 与train() eval()无关
                # torch.cuda.synchronize()
                # self.args.profiler[1].start()
                span_ner_mat_tensor, batch_span_ner_pred_lst, norm_link_scores = self.model.predict(inputs)
                # torch.cuda.synchronize()
                # self.args.profiler[1].end()

            # self.args.remained_spans_counter += len(batch_span_ner_pred_lst)
            # self.args.profiler[3].start()
            batch_span_lst_len = [(l + 1) * l // 2 for l in batch_seq_len.tolist()]
            batch_span_ner_pred_lst = utils.split_list(batch_span_ner_pred_lst.cpu().detach().numpy(), batch_span_lst_len)
            for exm, length, span_ner_pred_lst in zip(batch_ner_exm, batch_seq_len, batch_span_ner_pred_lst):
                if args.span_loss_type == 'softmax':
                    negative_set = {0}
                    exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst(span_ner_pred_lst, length, self.args.datareader.id2ent, negative_set=negative_set)  # softmax
                if args.span_loss_type == 'sigmoid':
                    exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst_sigmoid(span_ner_pred_lst, length, self.args.datareader.id2ent)  # sigmoid
            # self.args.profiler[3].end()
            # iterator.set_description(f'time1:{self.args.profiler[1].v} time2:{self.args.profiler[2].v} time3:{self.args.profiler[3].v}')

        exm_lst = data_loader.dataset.instances
        ep, er, ef, detail_info_str_e, *_ = utils.NerExample.eval(exm_lst, verbose=False, use_flat_pred_ent_dct=False)
        print(f'ori_metric: infer p-r-f: {ep:.3%}-{er:.3%}-{ef:.3%}')

        if args.flat:
            p, r, f, detail_info_str, *_ = utils.NerExample.eval(exm_lst, verbose=True, use_flat_pred_ent_dct=True)
            print(f'flat_metric: infer p-r-f: {p:.3%}-{r:.3%}-{f:.3%}')
        else:
            f = -1
        # self.args.reporter.append([self.args.link_threshold, f, ef])

    def fast_predict_span(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                  collate_fn=args.datareader.get_batcher_fn(gpu=self.use_gpu, mode=self.arch, device=self.device))
        print(f'infer num: {len(dataset)}')
        self.infer_num_steps = (len(dataset) - 1) // args.batch_size + 1

        self.run_epo = {
            'span': self.run_epo_span,
            'seq': self.run_epo_seq,
        }.get(self.arch, None)

        id2ent = self.args.datareader.id2ent

        # # ====profile
        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #                             schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        #                             on_trace_ready=torch.profiler.tensorboard_trace_handler('profiler_dir/predict2'),
        #                             # on_trace_ready=torch.profiler.tensorboard_trace_handler('profiler_dir/fast_predict'),
        #                             # on_trace_ready=torch.profiler.tensorboard_trace_handler('profiler_dir/fast_predict_new'),
        #                             profile_memory=True, record_shapes=True, with_stack=True) as prof:
        #     # t0 = time.time()
        #     # prof_loader = [batch for _, batch in zip(range(10), data_loader)]
        #     # logger.info(f"Data loading time: {time.time() - t0:.3f}s")
        #     # t0 = time.time()
        #     # for inputs in prof_loader:
        #     for step, inputs in enumerate(data_loader):
        #         if step >= (1 + 1 + 3) * 2: break
        #         # span_loss, span_ner_mat_tensor, batch_predict, conj_dot_product_score, f1, f1_detail = self.model(inputs)
        #         span_ner_mat_tensor, batch_span_ner_pred_lst, norm_link_scores = self.model.predict(inputs)
        #         # span_ner_mat_tensor, batch_span_ner_pred_lst, total_indices, norm_link_scores = self.model.fast_predict(inputs)
        #         # span_ner_mat_tensor, batch_span_ner_pred_lst, total_indices, norm_link_scores = self.model.fast_predict0825(inputs)
        #         prof.step()
        #     # logger.info(f"Model training time: {time.time() - t0:.3f}s")
        #
        # sort_by = "cuda_time_total" if self.use_gpu else "cpu_time_total"
        # prof_table = prof.key_averages().table(sort_by=sort_by, row_limit=200)
        # logger.info(f"\n{prof_table}")
        # ipdb.set_trace()
        # # ====

        self.model.eval()
        iterator = tqdm(data_loader, ncols=200) if self.use_tqdm else data_loader
        for num_steps, inputs in enumerate(iterator, start=1):
            batch_ner_exm = inputs['batch_ner_exm']

            if args.use_refine_mask: inputs['batch_refine_mask'] = None
            with torch.no_grad():  # 不计算梯度 更进一步节约内存 与train() eval()无关
                # torch.cuda.synchronize()
                # self.args.profiler[1].start()
                # span_ner_mat_tensor, batch_span_ner_pred_lst, norm_link_scores = self.model.predict(inputs)
                span_ner_mat_tensor, batch_span_ner_pred_lst, total_indices, norm_link_scores = self.model.fast_predict(inputs)
                # span_ner_mat_tensor, batch_span_ner_pred_lst, total_indices, norm_link_scores = self.model.fast_predict0825(inputs)
                # torch.cuda.synchronize()
                # self.args.profiler[1].end()
                # self.args.profiler[3].start()
                # # span_ner_mat_tensor, batch_span_ner_pred_lst, norm_link_scores = self.model.predict(inputs)
                # # torch.cuda.synchronize()
                # self.args.profiler[3].end()
                # print('mean perc:', sum(self.args.m_lst) / len(self.args.m_lst))
                # print('mean perc1:', sum(self.args.m_lst1) / len(self.args.m_lst1))

            # self.args.profiler[3].start()
            for exm in batch_ner_exm:
                exm.pred_ent_dct = {}
            batch_span_ner_pred_lst = batch_span_ner_pred_lst.detach().cpu().numpy()  # [*,ent]
            total_indices = total_indices.detach().cpu().numpy()  # [*,3] bsz, start, end
            # self.args.remained_spans_counter += len(total_indices)
            for idx, ent_idx in zip(*np.where(batch_span_ner_pred_lst >= 0.5)):
                bdx, s, e = total_indices[idx]
                prob = batch_span_ner_pred_lst[idx][ent_idx]
                batch_ner_exm[bdx].pred_ent_dct.setdefault(id2ent[ent_idx], []).append([s, e + 1, prob])
            # self.args.profiler[3].end()
            iterator.set_description(f'time1:{self.args.profiler[1].v} time2:{self.args.profiler[2].v} time3:{self.args.profiler[3].v}')

        exm_lst = data_loader.dataset.instances
        ep, er, ef, detail_info_str_e, *_ = utils.NerExample.eval(exm_lst, verbose=False, use_flat_pred_ent_dct=False)
        print(f'ori_metric: infer p-r-f: {ep:.3%}-{er:.3%}-{ef:.3%}')

        if args.flat:
            p, r, f, detail_info_str, *_ = utils.NerExample.eval(exm_lst, verbose=False, use_flat_pred_ent_dct=True)
            print(f'flat_metric: infer p-r-f: {p:.3%}-{r:.3%}-{f:.3%}')
        else:
            f = -1
        # self.args.reporter.append([self.args.link_threshold, f, ef])

    def predict_sents(self, sents):
        raw_exms = []
        for sent in sents:
            if isinstance(sent, str):
                exm = utils.NerExample(char_lst=list(sent), ent_dct={}, token_deli='')
                raw_exms.append(exm)
            elif isinstance(sent, dict):
                exm = utils.NerExample(char_lst=sent['text'], ent_dct={}, token_deli='')
                exm.file_name = 'dummy'
                raw_exms.append(exm)
            else:
                raise NotImplementedError
        if self.arch == 'span':
            return self.predict_span_sents(raw_exms)
        if self.arch == 'seq':
            return self.predict_seq_sents(raw_exms)

    def predict_span_sents(self, raw_exms):
        args = self.args

        sub_exms  = copy.deepcopy(raw_exms)
        [exm.process_ZHENG_by_tokenizer(args.datareader.tokenizer) for exm in sub_exms]

        dataset = args.datareader.build_dataset(sub_exms, max_len=512, lang='ZHENG', arch=args.arch)
        seg_info = dataset.seg_info
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                  collate_fn=args.datareader.get_batcher_fn(gpu=self.use_gpu, arch=self.arch, device=self.device))
        print(f'infer num: {len(dataset)}')
        self.infer_num_steps = (len(dataset) - 1) // args.batch_size + 1

        self.model.eval()
        iterator = tqdm(data_loader, ncols=200) if self.use_tqdm else data_loader
        for num_steps, inputs in enumerate(iterator, start=1):
            batch_ner_exm = inputs['batch_ner_exm']
            batch_seq_len = inputs['ori_seq_len']
            with torch.no_grad():  # 不计算梯度 更进一步节约内存 与train() eval()无关
                span_ner_mat_tensor, batch_span_ner_pred_lst, norm_link_scores = self.model.predict(inputs)
            batch_span_lst_len = [(l + 1) * l // 2 for l in batch_seq_len.tolist()]
            batch_span_ner_pred_lst = utils.split_list(batch_span_ner_pred_lst.cpu().detach().numpy(), batch_span_lst_len)
            for exm, length, span_ner_pred_lst in zip(batch_ner_exm, batch_seq_len, batch_span_ner_pred_lst):
                if args.span_loss_type == 'softmax':
                    negative_set = {0}
                    exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst(span_ner_pred_lst, length, args.datareader.id2ent, negative_set=negative_set)  # softmax
                if args.span_loss_type == 'sigmoid':
                    exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst_sigmoid(span_ner_pred_lst, length, args.datareader.id2ent)  # sigmoid

        splited_exms = dataset.instances
        # [print(exm) for exm in splited_exms]
        combined_exms = []
        for nums in seg_info:
            combined_exm = utils.NerExample.combine_exm(splited_exms[:nums])
            combined_exms.append(combined_exm)
            splited_exms = splited_exms[nums:]

        for c_exm, sub_exm, raw_exm in zip(combined_exms, sub_exms, raw_exms):
            # print(c_exm)
            # print(sub_exm)
            # print(raw_exm)
            raw_exm.char_lst = list(sub_exm.raw_text)
            raw_exm.pred_ent_dct = sub_exm.convert2raw_ent_dct(c_exm.pred_ent_dct)
            print(raw_exm)

        return [raw_exm.to_json_str(val_at_end=False, val_after_end=True, flat_pred_ent=True) for raw_exm in raw_exms]

    def predict_seq_sents(self, raw_exms):
        args = self.args
        dataset = args.datareader.build_dataset(raw_exms, lang='ZH', arch=args.arch)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                  collate_fn=args.datareader.get_batcher_fn(gpu=self.use_gpu, mode=self.arch, device=self.device))

        print(f'infer num: {len(dataset)}')
        self.infer_num_steps = (len(dataset) - 1) // args.batch_size + 1

        self.model.eval()
        iterator = tqdm(data_loader, ncols=200) if self.use_tqdm else data_loader
        for num_steps, inputs in enumerate(iterator, start=1):
            batch_ner_exm = inputs['batch_ner_exm']
            batch_seq_len = inputs['ori_seq_len']
            with torch.no_grad():  # 不计算梯度 更进一步节约内存 与train() eval()无关
                # torch.cuda.synchronize()
                _, emission, decode_ids = self.model(inputs, train=False)
                emission_probs = torch.softmax(emission, -1).cpu().detach().numpy()  # [bat,len,tag]

            for exm, decode_ids_, emission_probs_ in zip(batch_ner_exm, decode_ids, emission_probs):
                tag_lst = [args.datareader.id2tag[tag_id] for tag_id in decode_ids_]
                emission_prob = [emission_probs_[i, did] for i, did in enumerate(decode_ids_)]
                # assert len(tag_lst) == len(exm.char_lst)
                pred_ent_dct, _ = utils.NerExample.extract_entity_by_tags(tag_lst)
                for k, v_lst in pred_ent_dct.items():
                    for e in v_lst:
                        start, end = e[0], e[1]
                        mean_emission_prob = round(float(np.mean(emission_prob[start:end])), 4)
                        e.append(mean_emission_prob)  # 采用直接平均要素中所有token对应输出tag的发射概率
                        # e.append(1.)  # 假设概率为1
                exm.pred_ent_dct = pred_ent_dct

        for exm in raw_exms:
            print(exm)

        return [raw_exm.to_json_str(val_at_end=False, val_after_end=True, flat_pred_ent=True) for raw_exm in raw_exms]

if __name__ == "__main__":
    from rich.logging import RichHandler

    # handlers = [logging.StreamHandler(sys.stdout)]
    file_handler = logging.FileHandler(f"training.log", mode='a')
    file_handler.setFormatter(logging.Formatter(fmt='[%(asctime)s %(levelname)s] %(message)s',
                                                datefmt="%Y-%m-%d %H:%M:%S"))
    rich_handle = RichHandler(rich_tracebacks=True, )
    rich_handle.setFormatter(logging.Formatter(fmt='%(message)s',
                                               datefmt="%Y-%m-%d %H:%M:%S"))
    logging.basicConfig(level=logging.INFO,
                        format="%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[rich_handle, file_handler])

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser("QizNER")
    parser.add_argument('--info', help='information to distinguish model.', default='')
    parser.add_argument('--gpu', default='-2', type=str)  # '-1' use cpu, '-2' auto assign gpu

    # parser.add_argument('--pretrain_mode', default='feature_based')
    parser.add_argument('--pretrain_mode', default='fine_tuning')

    parser.add_argument('--arch', default='span')  # span based
    # parser.add_argument('--arch', default='seq')  # seq based

    # parser.add_argument('--interact_type', default='biaffine')
    parser.add_argument('--interact_type', default='self_attn')
    # parser.add_argument('--interact_type', default='add_attn')
    # parser.add_argument('--interact_type', default='sconcat')
    # parser.add_argument('--interact_type', default='cconcat')

    parser.add_argument('--span_loss_type', default='sigmoid')
    # parser.add_argument('--span_loss_type', default='softmax')

    # parser.add_argument('--use_slr', default=True, type=utils.str2bool)
    parser.add_argument('--use_slr', default=False, type=utils.str2bool)

    # parser.add_argument('--pooling_type', default='min')
    parser.add_argument('--pooling_type', default='softmin')

    parser.add_argument('--logsumexp_temp', default=0.3, type=float)

    # corpus:   # onto/conll03/ace04/ace05/genia
    # parser.add_argument('--corpus', default='ace05')
    # parser.add_argument('--corpus', default='ace04')
    parser.add_argument('--corpus', default='conll03')
    # parser.add_argument('--corpus', default='onto')
    # parser.add_argument('--corpus', default='genia')

    parser.add_argument('--eval_every_step', default=0, type=int)  # 250, 0 for plot convergence   # 0: not eval by step
    parser.add_argument('--eval_after_step', default=0, type=int)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--ckpt_dir', default=Path('model_ckpt'), type=Path)

    args = parser.parse_args()

    if args.gpu == '-1':
        args.device = torch.device('cpu')
    else:
        if args.gpu == '-2':
            args.device = utils.auto_device(logger, torch, np, subprocess)  # may be false to any GPUs
        else:
            args.device = torch.device(f'cuda:{args.gpu}')
    args.use_gpu = False
    if args.device.type.startswith('cuda'):
        torch.cuda.set_device(args.device)  # 不用受CUDA_VISIBLE_DEVICES限制 支持torch1.6 1.7
        args.use_gpu = True
    logger.info(f'truly used gpu: {args.device}')

    args.bert_model_dir = ['huggingface_model_resource/bert-large-cased',
                           'huggingface_model_resource/roberta-large',
                           'huggingface_model_resource/roberta-base',
                           ][2]
    args.dropout_rate = 0.2

    args.use_refine_mask = False

    if args.corpus == 'onto':
        """ontonote v4 标准5w"""
        args.flat = True
        ent_lst = ['ORDINAL', 'LOC', 'PRODUCT', 'NORP', 'WORK_OF_ART', 'LANGUAGE', 'GPE', 'TIME', 'PERCENT', 'MONEY', 'PERSON', 'CARDINAL', 'FAC', 'DATE', 'ORG', 'LAW', 'EVENT', 'QUANTITY']
        if args.span_loss_type == 'softmax': ent_lst = ['O'] + ent_lst
        datareader = NerDataReader(args.bert_model_dir, 512, ent_file_or_ent_lst=ent_lst, loss_type=args.span_loss_type, args=args)
        args.train_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/onto/train.jsonl', token_deli=' '), lang='ENG', arch=args.arch)
        args.dev_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/onto/dev.jsonl', token_deli=' '), lang='ENG', arch=args.arch)
        args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/onto/test.jsonl', token_deli=' '), lang='ENG', arch=args.arch)
        # args.trial_test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/onto/test.jsonl', token_deli=' ')[:5000], lang='ENG', arch=args.arch)

        if args.pretrain_mode == 'feature_based':
            pts = torch.load('corpora/onto/train.jsonl.pt')
            for exm, pt in zip(args.train_dataset.instances, pts):
                exm.pt = pt
            pts = torch.load('corpora/onto/dev.jsonl.pt')
            for exm, pt in zip(args.dev_dataset.instances, pts):
                exm.pt = pt
            pts = torch.load('corpora/onto/test.jsonl.pt')
            for exm, pt in zip(args.test_dataset.instances, pts):
                exm.pt = pt

    if args.corpus == 'conll03':
        """conll03"""
        args.flat = True
        ent_lst = ['ORG', 'MISC', 'PER', 'LOC']
        if args.span_loss_type == 'softmax': ent_lst = ['O'] + ent_lst
        datareader = NerDataReader(args.bert_model_dir, 512, ent_file_or_ent_lst=ent_lst, loss_type=args.span_loss_type, args=args)
        args.train_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/conll03/train_dev.jsonl', token_deli=' '), lang='ENG', arch=args.arch)
        args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/conll03/test.jsonl', token_deli=' '), lang='ENG', arch=args.arch)

        if args.pretrain_mode == 'feature_based':
            pts = torch.load('corpora/conll03/train_dev.jsonl.pt')
            for exm, pt in zip(args.train_dataset.instances, pts):
                exm.pt = pt
            pts = torch.load('corpora/conll03/test.jsonl.pt')
            for exm, pt in zip(args.test_dataset.instances, pts):
                exm.pt = pt

    if args.corpus == 'genia':
        """genia"""
        args.flat = False
        ent_lst = ['DNA', 'RNA', 'protein', 'cell_line', 'cell_type']
        if args.span_loss_type == 'softmax': ent_lst = ['O'] + ent_lst
        datareader = NerDataReader(args.bert_model_dir, 512, ent_file_or_ent_lst=ent_lst, loss_type=args.span_loss_type, args=args)

        args.train_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/genia/train.jsonl', token_deli=' '), lang='ENG', arch=args.arch)
        args.dev_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/genia/dev.jsonl', token_deli=' '), lang='ENG', arch=args.arch)
        args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/genia/test.jsonl', token_deli=' '), lang='ENG', arch=args.arch)

        if args.pretrain_mode == 'feature_based':
            pts = torch.load('corpora/genia/train.jsonl.pt')
            for exm, pt in zip(args.train_dataset.instances, pts):
                exm.pt = pt
            pts = torch.load('corpora/genia/dev.jsonl.pt')
            for exm, pt in zip(args.dev_dataset.instances, pts):
                exm.pt = pt
            pts = torch.load('corpora/genia/test.jsonl.pt')
            for exm, pt in zip(args.test_dataset.instances, pts):
                exm.pt = pt

    if args.corpus == 'ace04':
        """ace04"""
        args.flat = False
        ent_lst = ['LOC', 'WEA', 'GPE', 'PER', 'FAC', 'ORG', 'VEH']
        if args.span_loss_type == 'softmax': ent_lst = ['O'] + ent_lst
        datareader = NerDataReader(args.bert_model_dir, 512, ent_file_or_ent_lst=ent_lst, loss_type=args.span_loss_type, args=args)
        args.train_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/ace04/train.jsonl', token_deli=' '), lang='ENG', arch=args.arch)
        args.dev_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/ace04/dev.jsonl', token_deli=' '), lang='ENG', arch=args.arch)
        args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/ace04/test.jsonl', token_deli=' '), lang='ENG', arch=args.arch)

        if args.pretrain_mode == 'feature_based':
            pts = torch.load('corpora/ace04/train.jsonl.pt')
            for exm, pt in zip(args.train_dataset.instances, pts):
                exm.pt = pt
            pts = torch.load('corpora/ace04/dev.jsonl.pt')
            for exm, pt in zip(args.dev_dataset.instances, pts):
                exm.pt = pt
            pts = torch.load('corpora/ace04/test.jsonl.pt')
            for exm, pt in zip(args.test_dataset.instances, pts):
                exm.pt = pt

    if args.corpus == 'ace05':
        """ace05"""
        args.flat = False
        ent_lst = ['LOC', 'WEA', 'GPE', 'PER', 'FAC', 'ORG', 'VEH']
        if args.span_loss_type == 'softmax': ent_lst = ['O'] + ent_lst
        datareader = NerDataReader(args.bert_model_dir, 512, ent_file_or_ent_lst=ent_lst, loss_type=args.span_loss_type, args=args)
        args.train_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/ace05/train.jsonl', token_deli=' '), lang='ENG', arch=args.arch)
        args.dev_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/ace05/dev.jsonl', token_deli=' '), lang='ENG', arch=args.arch)
        args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/ace05/test.jsonl', token_deli=' '), lang='ENG', arch=args.arch)

        if args.pretrain_mode == 'feature_based':
            pts = torch.load('corpora/ace05/train.jsonl.pt')
            for exm, pt in zip(args.train_dataset.instances, pts):
                exm.pt = pt
            pts = torch.load('corpora/ace05/dev.jsonl.pt')
            for exm, pt in zip(args.dev_dataset.instances, pts):
                exm.pt = pt
            pts = torch.load('corpora/ace05/test.jsonl.pt')
            for exm, pt in zip(args.test_dataset.instances, pts):
                exm.pt = pt

    if args.corpus == 'cluener':
        """cluener"""
        args.bert_model_dir = 'huggingface_model_resource/chinese-roberta-wwm-ext'
        args.flat = True
        ent_lst = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
        if args.span_loss_type == 'softmax': ent_lst = ['O'] + ent_lst
        datareader = NerDataReader(args.bert_model_dir, 512, ent_file_or_ent_lst=ent_lst, loss_type=args.span_loss_type, args=args)
        args.train_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/cluener/train.jsonl', token_deli=''), lang='ZH', arch=args.arch)
        args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/cluener/dev.jsonl', token_deli=''), lang='ZH', arch=args.arch)

    if args.corpus == 'general_NER_predictor_by_cluener':
        """cluener"""
        args.bert_model_dir = 'huggingface_model_resource/FinBERT_L-12_H-768_A-12_pytorch'
        args.bert_model_dir = 'huggingface_model_resource/mengzi-bert-base-fin'
        args.flat = True
        ent_lst = [
            'address',
            # 'book',
            # 'company',
            # 'game',
            # 'government',
            # 'movie',
            'name',
            'organization',
            # 'position',
            # 'scene',
        ]
        ent2ent_map = {
            'company': 'organization',
            'government': 'organization',
            'scene': 'address',
        }
        keep_ent_lst = ['organization', 'name', 'address']
        datareader = NerDataReader(args.bert_model_dir, 512, ent_file_or_ent_lst=ent_lst, loss_type=args.span_loss_type, args=args)
        train_exm_lst = utils.NerExample.load_from_jsonl('../data/cluener_public/train.jsonl', token_deli='')
        dev_exm_lst = utils.NerExample.load_from_jsonl('../data/cluener_public/dev.jsonl', token_deli='')
        for exm in train_exm_lst:
            exm.ent_type_convert(ent2ent_map=ent2ent_map)
            exm.remove_ent_by_type(ent_type_lst=keep_ent_lst, input_keep=True)
        for exm in dev_exm_lst:
            exm.ent_type_convert(ent2ent_map=ent2ent_map)
            exm.remove_ent_by_type(ent_type_lst=keep_ent_lst, input_keep=True)
        args.train_dataset = datareader.build_dataset(train_exm_lst, lang='ZH', arch=args.arch)
        args.test_dataset = datareader.build_dataset(dev_exm_lst, lang='ZH', arch=args.arch)
        # args.train_dataset = datareader.build_dataset(train_exm_lst, lang='ENG', arch=args.arch)
        # args.test_dataset = datareader.build_dataset(dev_exm_lst, lang='ENG', arch=args.arch)

    if args.corpus == 'huawei':
        """huawei"""
        args.bert_model_dir = 'huggingface_model_resource/bert-base-multilingual-cased'
        args.bert_model_dir = 'huggingface_model_resource/mengzi-bert-base-fin'
        args.bert_model_dir = 'huggingface_model_resource/roberta-base-finetuned-cluener2020-chinese'
        args.bert_model_dir = 'huggingface_model_resource/chinese-roberta-wwm-ext'
        args.flat = True
        ent_lst = ['ENT1', 'ENT2']
        if args.span_loss_type == 'softmax': ent_lst = ['O'] + ent_lst
        datareader = NerDataReader(args.bert_model_dir, 512, ent_file_or_ent_lst=ent_lst, loss_type=args.span_loss_type, args=args)
        args.train_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('', token_deli=''), lang='ZHENG', arch=args.arch)
        args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('', token_deli=''), lang='ZHENG', arch=args.arch)

    if args.corpus == 'event':
        """ test """
        args.bert_model_dir = 'huggingface_model_resource/bert-base-multilingual-cased'
        # args.bert_model_dir = 'huggingface_model_resource/mengzi-bert-base-fin'
        # args.bert_model_dir = 'huggingface_model_resource/roberta-base-finetuned-cluener2020-chinese'
        # args.bert_model_dir = 'huggingface_model_resource/chinese-roberta-wwm-ext'
        # args.bert_model_dir = 'huggingface_model_resource/nghuyong/ernie-3.0-base-zh'

        exm_lst = utils.NerExample.load_from_jsonl('tmp_test/long_text.jsonl', token_deli='')
        args.flat = True
        ent_lst = utils.NerExample.get_ents_set(exm_lst)
        if args.span_loss_type == 'softmax': ent_lst = ['O'] + ent_lst
        datareader = NerDataReader(args.bert_model_dir, 512, ent_file_or_ent_lst=ent_lst, loss_type=args.span_loss_type, args=args)
        # args.train_dataset = datareader.build_dataset(exm_lst[:-500], lang='ZHENG', arch=args.arch)
        # args.test_dataset = datareader.build_dataset(exm_lst[-500:], lang='ZHENG', arch=args.arch)

        args.train_dataset = datareader.build_dataset(exm_lst[:-500], lang='ZHENG', arch=args.arch,
                                                      max_len=64, prefix_context_len=16, neg_ratio=1., cached_file='tmp_test/long_text_train.jsonl')
        args.test_dataset = datareader.build_dataset(exm_lst[-500:], lang='ZHENG', arch=args.arch,
                                                     max_len=64, prefix_context_len=16, cached_file='tmp_test/long_text_test.jsonl')

    if args.corpus == 'event_10fold':
        args.bert_model_dir = 'huggingface_model_resource/bert-base-multilingual-cased'
        exm_lst = utils.NerExample.load_from_jsonl('tmp_test/long_text.jsonl', token_deli='')
        args.flat = True
        ent_lst = utils.NerExample.get_ents_set(exm_lst)
        if args.span_loss_type == 'softmax': ent_lst = ['O'] + ent_lst
        datareader = NerDataReader(args.bert_model_dir, 512, ent_file_or_ent_lst=ent_lst, loss_type=args.span_loss_type, args=args)

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=10, shuffle=True)

        for kid, (train_index, test_index) in enumerate(kf.split(exm_lst)):
            curr_kf_train_exm_lst = [exm for idx, exm in enumerate(exm_lst) if idx in train_index]
            curr_kf_test_exm_lst = [exm for idx, exm in enumerate(exm_lst) if idx in test_index]
            args.train_dataset = datareader.build_dataset(curr_kf_train_exm_lst, lang='ZHENG', arch=args.arch,
                                                          max_len=64, prefix_context_len=16, neg_ratio=1.)
            args.test_dataset = datareader.build_dataset(curr_kf_test_exm_lst, lang='ZHENG', arch=args.arch)

            args.datareader = datareader
            # args.batch_size = 16
            # args.batch_size = 1
            args.num_warmup_steps = 1000
            args.num_warmup_steps = 722  # 2epo
            # args.num_warmup_steps = 3610
            args.lr = [1e-5, 2e-5, 8e-6][0]
            if args.interact_type == 'cconcat':
                args.batch_size = 12  # to avoid out of GPU memory
            if args.corpus in ['ace04', 'ace05']:
                args.batch_size = 12
            if args.pretrain_mode == 'feature_based':
                args.batch_size = 256
                args.batch_size = 64
                args.lr = 1e-3
                args.num_warmup_steps = 100

            logger.info(f'using corpus {args.corpus}')

            # ====Train====
            logger.info(utils.header_format(f"Starting K_Fold {kid}", sep='='))
            args.ckpt_dir = f'model_ckpt/KFold{kid}'
            trainer = Trainer(args, arch=args.arch)
            logger.info(utils.header_format(f"Training K_Fold {kid}", sep='='))
            with ipdb.launch_ipdb_on_exception():
                trainer.train()
        exit(0)




    if args.use_refine_mask:
        for name in ['train_dataset', 'dev_dataset', 'test_dataset']:
            if hasattr(args, name):
                for exm in getattr(args, name).instances:
                    exm.refine_mask = utils.get_refined_score_mask(exm)
    args.datareader = datareader
    # args.batch_size = 16
    # args.batch_size = 1
    args.num_warmup_steps = 1000
    args.num_warmup_steps = 722  # 2epo
    # args.num_warmup_steps = 3610
    args.lr = [1e-5, 2e-5, 8e-6][0]
    if args.interact_type == 'cconcat':
        args.batch_size = 12  # to avoid out of GPU memory
    if args.corpus in ['ace04', 'ace05']:
        args.batch_size = 12
    if args.pretrain_mode == 'feature_based':
        args.batch_size = 256
        args.batch_size = 64
        args.lr = 1e-3
        args.num_warmup_steps = 100

    logger.info(f'using corpus {args.corpus}')

    # ====Train====
    logger.info(utils.header_format("Starting", sep='='))
    trainer = Trainer(args, arch=args.arch)
    logger.info(utils.header_format("Training", sep='='))
    with ipdb.launch_ipdb_on_exception():
        trainer.train()
    exit(0)

    # ====Evaluation====
    logger.info(utils.header_format("Starting", sep='='))
    exist_ckpt = 'model_ckpt/22_11_07_17_31_conll03_span_self_attn_sigmoid_SLR_softmin/best_test_model_24_8664.ckpt'
    trainer = Trainer(args, arch=args.arch, exist_ckpt=exist_ckpt, evaluate=True)
    logger.info(utils.header_format("Evaluation_by_existed_ckpt", sep='='))
    with ipdb.launch_ipdb_on_exception():
        trainer.train()
    exit(0)

    # ====Inference====
    logger.info(utils.header_format("Starting", sep='='))
    exist_ckpt = 'model_ckpt/23_03_02_22_34_event_span_self_attn_sigmoid/best_test_model_3_2349.ckpt'
    args = utils.load_args_by_json_file(Path(exist_ckpt).parent / 'args.json', exist_args=args)
    trainer = Trainer(args, arch=args.arch, exist_ckpt=exist_ckpt, evaluate=True)
    logger.info(utils.header_format("Inference_by_existed_ckpt", sep='='))
    args.infer_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('corpora/conll03/test.jsonl', token_deli=' '), arch=args.arch)
    args.link_threshold = -2  # onto
    with ipdb.launch_ipdb_on_exception():
        trainer.predict_span(args.infer_dataset)
        trainer.fast_predict_span(args.infer_dataset)
    exit(0)

    # ====inference sents====
    logger.info(utils.header_format("Starting", sep='='))
    exist_ckpt = 'model_ckpt/23_03_02_22_34_event_span_self_attn_sigmoid/best_test_model_3_2349.ckpt'
    args = utils.load_args_by_json_file(Path(exist_ckpt).parent / 'args.json', exist_args=args)
    trainer = Trainer(args, arch=args.arch, exist_ckpt=exist_ckpt, evaluate=True)
    logger.info(utils.header_format("Inference_by_existed_ckpt", sep='='))
    sents = [e['text'] for e in utils.load_jsonl('tmp_test/long_text.jsonl')[:10]]
    trainer.predict_sents(sents)
    exit(0)

    # """ predict test speed """
    # exist_ckpt = 'model_ckpt/22_06_19_14_53_onto_span_biaffine_sigmoid_SLR_softmin/best_dev_model_5_15000.ckpt'  #
    # args = utils.load_args_by_json_file(Path(exist_ckpt).parent / 'args.json', exist_args=args)
    # # args.test_dataset = args.trial_test_dataset
    # # args.batch_size = 1
    # args.batch_size = [128, 256, 368][0]
    # trainer = Trainer(args, arch=args.arch, exist_ckpt=exist_ckpt, evaluate=True)
    # args.profiler = utils.Time_Meaners(5, start_with=1)
    # # print('\n===predict===')
    # # trainer.predict(args.test_dataset)  # warm-up
    # # args.profiler.reset()
    # args.reporter = []
    # args.remained_spans_counter = 0
    #
    # # print('\n===fast_predict===')
    # # for i in [-1000000] + list(range(-30, 20, 1)) + [20]:
    # # # for i in [-1000000]:
    # #     args.link_threshold = i
    # #     print('args.link_threshold', args.link_threshold)
    # #     trainer.fast_predict(args.test_dataset)  # warm-up
    # #     # trainer.predict(args.test_dataset)  # warm-up
    # #     # snum = args.remained_spans_counter / 8262
    # #     # snum = args.remained_spans_counter / 711  # ace05
    # #     snum = args.remained_spans_counter / 620  # ace04
    # #     args.reporter[-1].append(args.remained_spans_counter)
    # #     args.reporter[-1].append(snum)
    # #     print(*args.reporter, sep='\n')
    # #     utils.save_jsonl([{'threshold': t, 'f':f, 'ef': ef, 'snum': snum,'total_num_remained':tnum, } for t, f, ef, tnum, snum in args.reporter], Path(exist_ckpt).parent / 'th_num_col.jsonl')
    # #     args.remained_spans_counter = 0
    # # exit(0)
    #
    # args.m_gather = 0
    # args.m_fast_inter = 0
    # args.m_inter = 0
    # args.m_lst = []
    # args.m_lst1 = []
    #
    # args.profiler.reset()
    # args.link_threshold = -2  # onto
    # trainer.fast_predict_span(args.test_dataset)  # warm-up
    # args.profiler.reset()
    #
    # # exit(0)
    # print('\n===predict===')
    # trainer.predict_span(args.test_dataset)
    # args.profiler.reset()
    #
    # print('\n===fast_predict===')
    # trainer.fast_predict_span(args.test_dataset)
    # args.profiler.reset()
    #
    # print('\n===predict===')
    # trainer.predict_span(args.test_dataset)
    # args.profiler.reset()
    #
    # print('\n===fast_predict===')
    # trainer.fast_predict_span(args.test_dataset)
