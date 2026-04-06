from distutils.command.config import config
import os
import torch
import numpy as np
from utility.load_data import *  # 确保包含 Data_for_BPR 类
import pandas as pd
import sys
from time import time
from sklearn.metrics import roc_auc_score
import random
# 导入你的 BPR 模型（匹配实际接口）
from Model.BPR import BPR
from utility.compute import *  # 使用修改后的 compute.py（兼容 BPR）


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.embed_size = 48
        self.regs = 0
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = 'Data/Process/'
        self.dataset = 'BookCrossing'
        self.attack = '0.02'
        self.data_type = 'full'
        self.seed = 1024
        self.init_std = 1e-4

    def reset(self, config):
        for name, val in config.items():
            setattr(self, name, val)


class early_stoper(object):
    def __init__(self, refer_metric='valid_auc', stop_condition=10):
        super().__init__()
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.stop_condition = stop_condition
        self.init_flag = True
        self.refer_metric = refer_metric

    def update_and_isbest(self, eval_metric, epoch):
        if self.init_flag:
            self.best_epoch = epoch
            self.init_flag = False
            self.best_eval_result = eval_metric
            return True
        elif eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric]:
            self.best_eval_result = eval_metric
            self.not_change = 0
            self.best_epoch = epoch
            return True
        else:
            self.not_change += 1
            return False

    def is_stop(self):
        return self.not_change > self.stop_condition


def main(config_args):
    args = model_hyparameters()
    args.reset(config_args)

    # 固定随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    save_name = 'BPR_'
    for name_str, name_val in config_args.items():
        save_name += name_str + '-' + str(name_val) + '-'
    weights_dir = './Weights/BPR/'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    save_path = weights_dir + save_name + "m.pth"


    data_generator = Data_for_BPR(
        data_path=args.data_path + args.dataset + '/' + args.attack,
        batch_size=args.batch_size
    )
    data_generator.set_train_mode(args.data_type)


    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    model = BPR(data_config=config, args=args).cuda()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)


    best_epoch = 0
    best_valid_auc = 0
    best_test_auc = 0
    e_stoper = early_stoper(refer_metric='valid_auc', stop_condition=10)
    mask = get_eval_mask(data_generator)  # 兼容BPR的compute.py


    total_train_start_time = time()

    cumulative_train_time = 0.0
    cumulative_eval_time = 0.0


    for epoch in range(args.epoch):
        t1 = time()
        loss, bpr_loss, reg_loss = 0., 0., 0.


        for batch_data in data_generator.batch_generator():

            users = batch_data[:, 0].cuda().long()
            pos_items = batch_data[:, 1].cuda().long()
            neg_items = batch_data[:, 2].cuda().long()


            batch_bpr_loss, batch_reg_loss, batch_total_loss = model.train_one_batch(users, pos_items, neg_items, opt)


            loss += batch_total_loss
            bpr_loss += batch_bpr_loss
            reg_loss += batch_reg_loss


        if np.isnan(loss):
            print('ERROR: loss is nan.')
            sys.exit()


        t2 = time()
        valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and = get_eval_result_original(
            data_generator, model, mask)


        epoch_train_time = t2 - t1
        epoch_eval_time = time() - t2
        cumulative_train_time += epoch_train_time
        cumulative_eval_time += epoch_eval_time


        perf_str = (
            f"epoch: {epoch}, train_time: {epoch_train_time:.6f}, eval_time: {epoch_eval_time:.6f}, "
            f"train_loss: {loss:.6f}, bpr_loss: {bpr_loss:.6f}, reg_loss: {reg_loss:.6f}, "
            f"valid auc: {valid_auc:.6f}, valid auc or: {valid_auc_or:.6f}, valid auc and: {valid_auc_and:.6f}, "
            f"test auc: {test_auc:.6f}, test auc or: {test_auc_or:.6f}, test auc and: {test_auc_and:.6f}"
        )
        print(perf_str)


        one_result = {'valid_auc': valid_auc, 'test_auc': test_auc}
        is_best = e_stoper.update_and_isbest(one_result, epoch)
        if is_best:
            best_epoch = epoch
            best_valid_auc = valid_auc
            best_test_auc = test_auc
            torch.save(model.state_dict(), save_path)
            print(f"saving the best model to {save_path}")


        if e_stoper.is_stop():
            print(f"Early stop at epoch {epoch}, best model saved to {save_path}")
            break


    total_train_end_time = time()
    total_elapsed_time = total_train_end_time - total_train_start_time


    final_perf = (
        f'best_epoch = {best_epoch}, best_valid_auc = {best_valid_auc:.6f}, best_test_auc = {best_test_auc:.6f}\n'
        f'all train: {cumulative_train_time:.6f} 秒\n'
        f'all eval: {cumulative_eval_time:.6f} 秒\n'
        f'all: {total_elapsed_time:.6f} 秒'
    )
    print(final_perf)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = {
        'lr': 1e-3,  # [1e-2, 1e-3, 1e-4]
        'embed_size': 48,  # [32, 48, 64]
        'batch_size': 2048,
        'data_type': 'full',
        'dataset': 'Amazon',  #[BookCrossing, Amazon,Mooccube]
        'attack':'0.02',  # [0.02, 0.01]
        'seed': 1024,
        'init_std': 1e-3  # [1e-2, 1e-3, 1e-4]
    }
    main(config)