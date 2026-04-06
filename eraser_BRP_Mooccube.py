import numpy as np
from utility.load_data import *
import os
import sys
import pickle
import torch
from time import time, perf_counter
import random

from Model.Eraser import RecEraser_BPR
from utility.compute import *


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.regs = 0
        self.regs_agg = 0
        self.embed_size = 48
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = 'Data/Process/'
        self.dataset = 'BookCrossing'
        self.attack = '0.02'
        self.verbose = 1
        self.data_type = 'full'
        self.save_flag = 1
        self.drop_prob = 0
        self.biased = False
        self.init_std = 1e-3
        self.part_type = 1
        self.part_num = 10
        self.part_T = 50
        self.seed = 1024

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
        else:
            if eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric]:
                self.best_eval_result = eval_metric
                self.not_change = 0
                self.best_epoch = epoch
                return True
            else:
                self.not_change += 1
                return False

    def is_stop(self):
        return self.not_change > self.stop_condition

    def re_init(self, stop_condition=None):
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.init_flag = True
        if stop_condition is not None:
            self.stop_condition = stop_condition


def main(config_args):
    args = model_hyparameters()
    args.reset(config_args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = {'n_users': 0, 'n_items': 0}
    data_generator = Data_for_RecEraser_BPR(
        args.data_path + args.dataset + '/' + args.attack,
        args.batch_size, args.part_type, args.part_num, args.part_T
    )
    data_generator.set_train_mode(args.data_type)
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    total_time = [0] * (args.part_num + 1)
    model = RecEraser_BPR(config, args).cuda()
    e_stoper = early_stoper(refer_metric='valid_auc', stop_condition=10)

    opt_list = [torch.optim.Adam(model.parameters(), lr=args.lr) for _ in range(args.part_num)]
    opt_list.append(torch.optim.Adam(model.parameters(), lr=args.lr))

    valid_data = data_generator.valid[['user', 'item', 'label']].values
    test_data = data_generator.test[['user', 'item', 'label']].values
    mask = get_eval_mask(data_generator)

    weights_save_path = ''
    if args.save_flag == 1:
        weights_save_path = './Weights/BPR_Eraser/bpr_'
        for name_, val_ in config_args.items():
            weights_save_path += name_ + '_' + str(val_) + '_'
        ensureDir(weights_save_path)

    # =========================
    # 1. 训练局部模型
    # =========================
    for i in range(args.part_num):
        e_stoper.re_init()
        weights_save_path_local = weights_save_path + "-local-" + str(i) + ".pk"
        print(f"start training {i}-th sub-model")

        for epoch in range(args.epoch):
            t1 = time()
            loss, bpr_loss, reg_loss = 0.0, 0.0, 0.0
            model.train()

            train_loader_i = data_generator.batch_generator_local(local_id=i)

            for batch_data in train_loader_i:
                users = batch_data[:, 0].cuda().long()
                pos_items = batch_data[:, 1].cuda().long()
                neg_items = batch_data[:, 2].cuda().long()

                opt_list[i].zero_grad()
                batch_loss, batch_bpr_loss, batch_reg_loss, _ = model.single_model(users, pos_items, neg_items, i)

                if torch.isnan(batch_loss).any():
                    print('ERROR: batch_loss is nan.')
                    sys.exit()

                batch_loss.backward()
                opt_list[i].step()

                loss += batch_loss.item()
                bpr_loss += batch_bpr_loss.item()
                reg_loss += batch_reg_loss.item()

            t2 = time()
            with torch.no_grad():
                valid_predictions = model.single_prediction(valid_data[:, 0], valid_data[:, 1], i)
                if isinstance(valid_predictions, torch.Tensor):
                    valid_predictions = valid_predictions.detach().cpu().numpy()

                test_predictions = model.single_prediction(test_data[:, 0], test_data[:, 1], i)
                if isinstance(test_predictions, torch.Tensor):
                    test_predictions = test_predictions.detach().cpu().numpy()

                valid_auc = safe_auc(valid_data[:, -1], valid_predictions)
                test_auc = safe_auc(test_data[:, -1], test_predictions)

            one_result = {'valid_auc': valid_auc, 'test_auc': test_auc}

            t3 = time()
            if args.verbose > 0:
                perf_str = (
                    f'[local_model {i}] epoch {epoch} '
                    f'[{t2 - t1:.4f}s + {t3 - t2:.4f}s]: '
                    f'train_loss=[{loss:.4f}={bpr_loss:.4f}+{reg_loss:.4f}], '
                    f'[valid,test] auc=[{valid_auc:.4f}, {test_auc:.4f}]'
                )
                print(perf_str)

            is_best = e_stoper.update_and_isbest(one_result, epoch)
            if is_best:
                total_time[i] += (t2 - t1)
                if args.save_flag == 1:
                    user_emb = model.user_embedding.weight[
                               :, i * model.emb_dim:(i + 1) * model.emb_dim
                               ].detach().cpu().numpy()
                    item_emb = model.item_embedding.weight[
                               :, i * model.emb_dim:(i + 1) * model.emb_dim
                               ].detach().cpu().numpy()

                    with open(weights_save_path_local, 'wb') as f:
                        pickle.dump((user_emb, item_emb), f)

            if e_stoper.is_stop():
                print(f"local model {i} best epoch: {e_stoper.best_epoch}")
                break

    # =========================
    # 2. 加载局部模型权重
    # =========================
    if args.save_flag == 1:
        user_emb = []
        item_emb = []
        for i in range(args.part_num):
            weights_save_path_local = weights_save_path + "-local-" + str(i) + ".pk"
            with open(weights_save_path_local, 'rb') as f:
                emb1, emb2 = pickle.load(f)
                user_emb.append(emb1)
                item_emb.append(emb2)

        user_emb = np.concatenate(user_emb, axis=-1)
        item_emb = np.concatenate(item_emb, axis=-1)
        user_emb = torch.from_numpy(user_emb).float().cuda()
        item_emb = torch.from_numpy(item_emb).float().cuda()

        with torch.no_grad():
            model.user_embedding.weight.copy_(user_emb.data)
            model.item_embedding.weight.copy_(item_emb.data)

    # =========================
    # 3. 训练聚合模型
    # =========================
    e_stoper.re_init()
    for epoch in range(args.epoch):
        t1 = time()
        loss, bpr_loss, reg_loss = 0.0, 0.0, 0.0
        model.train()

        agg_train_loader = data_generator.batch_generator()

        for batch_data in agg_train_loader:
            users = batch_data[:, 0].cuda().long()
            pos_items = batch_data[:, 1].cuda().long()
            neg_items = batch_data[:, 2].cuda().long()

            opt_list[-1].zero_grad()
            batch_loss, batch_bpr_loss, batch_reg_loss, _, _, _ = model.compute_agg_model(users, pos_items, neg_items)

            if torch.isnan(batch_loss).any():
                print('ERROR: batch_loss is nan.')
                sys.exit()

            batch_loss.backward()
            opt_list[-1].step()

            loss += batch_loss.item()
            bpr_loss += batch_bpr_loss.item()
            reg_loss += batch_reg_loss.item()

        t2 = time()
        valid_auc, valid_auc_or, valid_auc_and, \
        test_auc, test_auc_or, test_auc_and = get_eval_result(
            data_generator, model, mask
        )

        one_result = {'valid_auc': valid_auc, 'test_auc': test_auc}

        t3 = time()
        if args.verbose > 0:
            print(
                "epoch: %d, train_time: %.6f, eval_time: %.6f, "
                "valid auc:%.6f, valid auc or:%.6f, valid auc and:%.6f, "
                "test auc:%.6f, test auc or:%.6f, test auc and:%.6f"
                % (
                    epoch, t2 - t1, t3 - t2,
                    valid_auc, valid_auc_or, valid_auc_and,
                    test_auc, test_auc_or, test_auc_and
                )
            )

        is_best = e_stoper.update_and_isbest(one_result, epoch)
        if is_best:
            total_time[-1] += (t2 - t1)
            if args.save_flag == 1:
                torch.save(model.state_dict(), weights_save_path + "-m.pth")

        if e_stoper.is_stop():
            print("best epoch:", e_stoper.best_epoch)
            break

    # =========================
    # 4. 训练时间总结
    # =========================
    print("\nTraining Time Summary:")
    for i in range(args.part_num):
        print(f"local{i + 1}: {total_time[i]:.6f} seconds")
    local_sum = sum(total_time[:-1])
    print(f"local_sum: {local_sum:.6f} seconds")
    print(f"agg: {total_time[-1]:.6f} seconds")
    total_sum = sum(total_time)
    print(f"total: {total_sum:.6f} seconds")


if __name__ == '__main__':
    begin = perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    config = {
        'lr': 1e-3,
        'embed_size': 32,
        'batch_size': 2048,
        'data_type': 'full',
        'init_std': 1e-4,
        'dataset': 'Mooccube',
        'attack': '0.02',
        'seed': 1024,
        'part_type': 1,# 0: whole data, 1: interaction_based, 2: user_based, 3: random  4:HDRF  5:BCESP 6:CESP
    }

    main(config)
    end = perf_counter()
    print("ALL Time taken:", (end - begin), file=sys.stderr)