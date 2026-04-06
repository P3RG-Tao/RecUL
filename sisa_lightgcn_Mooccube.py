import numpy as np
from utility.load_data import *
import os
import sys
import pickle
import torch
from sklearn.metrics import roc_auc_score
from time import time
import random
from Model.Eraser import RecEraser_LightGCN
from utility.compute import *
from time import perf_counter


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
        self.pretrain = 0
        self.n_layers = 1
        self.verbose = 1
        self.data_type = 'full'
        self.save_flag = 1
        self.drop_prob = 0
        self.biased = False
        self.init_std = 1e-3
        self.part_type = 1  # 0: whole data, 1: interaction_based, 2: user_based, 3: random
        self.part_num = 10  # partition number
        self.part_T = 50  # iteraction for partition
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
        if self.not_change > self.stop_condition:
            return True
        else:
            return False

    def re_init(self, stop_condition=None):
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.init_flag = True
        if stop_condition is not None:
            self.stop_condition = stop_condition


def main(config_args):
    args = model_hyparameters()
    assert config_args is not None
    args.reset(config_args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = dict()
    data_generator = Data_for_RecEraser_LightGCN(args.data_path + args.dataset + '/' + args.attack, args.batch_size, args.part_type, args.part_num, args.part_T)
    data_generator.set_train_mode(args.data_type)
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    total_time = [0] * args.part_num  # 移除 agg 时间位

    model = RecEraser_LightGCN(config, args).cuda()
    model.Graph = data_generator.Graph
    e_stoper = early_stoper(refer_metric='valid_auc', stop_condition=10)

    # 只保留子模型优化器，移除聚合优化器
    opt_list = [torch.optim.Adam(model.parameters(), lr=args.lr) for i in range(args.part_num)]

    valid_data = data_generator.valid[['user', 'item', 'label']].values
    test_data = data_generator.test[['user', 'item', 'label']].values
    mask = get_eval_mask(data_generator)

    if args.save_flag == 1:
        weights_save_path = './Weights/LightGCN_SISA/lightgcn_'
        for name_, val_ in config_args.items():
            weights_save_path += name_ + '_' + str(val_) + '_'
        ensureDir(weights_save_path)

    # ===================== 子模型训练（完全不变）=====================
    for i in range(args.part_num):
        e_stoper.re_init()
        weights_save_path_local = weights_save_path + "-local-" + str(i) + ".pk"

        print("start training %d-th sub-model" % (i))
        train_loader_i = data_generator.batch_generator_local(local_id=i)
        for epoch in range(args.epoch):
            t1 = time()
            loss, bce_loss, reg_loss = 0., 0., 0.
            model.train()
            for batch_data in train_loader_i:
                users, items, labels = batch_data[:, 0].cuda().long(), batch_data[:, 1].cuda().long(), batch_data[:, 2].cuda().float()
                model.zero_grad()
                batch_loss, batch_bce_loss, batch_reg_loss, _ = model.single_model(users, items, labels, i)
                batch_loss.backward()
                opt_list[i].step()
                loss += batch_loss
                bce_loss += batch_bce_loss
                reg_loss += batch_reg_loss

            if torch.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            t2 = time()
            valid_predictions = model.single_prediction(valid_data[:, 0], valid_data[:, 1], i)
            valid_auc = roc_auc_score(valid_data[:, -1], valid_predictions)
            test_predictions = model.single_prediction(test_data[:, 0], test_data[:, 1], i)
            test_auc = roc_auc_score(test_data[:, -1], test_predictions)
            one_result = {'valid_auc': valid_auc, 'test_auc': test_auc}

            t3 = time()
            if args.verbose > 0:
                perf_str = '[local_model %d] epoch %d [%.4fs + %.4fs]: train_loss=[%.4f=%.4f + %.4f], [valid,test] auc=[%.4f, %.4f]' \
                            % \
                           (i, epoch, t2 - t1, t3 - t2, loss, bce_loss, reg_loss, valid_auc, test_auc)
                print(perf_str)

            is_best = e_stoper.update_and_isbest(one_result, epoch)
            if is_best:
                total_time[i] += (t2-t1)
                user_emb = model.user_embedding.weight[:, i * (model.emb_dim):(i + 1) * model.emb_dim].detach().cpu().numpy()
                item_emb = model.item_embedding.weight[:, i * (model.emb_dim):(i + 1) * model.emb_dim].detach().cpu().numpy()
                with open(weights_save_path_local, 'wb') as f:
                    pickle.dump((user_emb, item_emb), f)

            if e_stoper.is_stop():
                break

    # ===================== 加载所有子模型 =====================
    user_emb_list = []
    item_emb_list = []
    for i in range(args.part_num):
        weights_save_path_local = weights_save_path + "-local-" + str(i) + ".pk"
        with open(weights_save_path_local, 'rb') as f:
            emb1, emb2 = pickle.load(f)
            user_emb_list.append(emb1)
            item_emb_list.append(emb2)

    # ===================== 标准 SISA 聚合：直接平均 =====================
    print("\n=== 标准 SISA 聚合推理（所有子模型输出直接平均）===")
    with torch.no_grad():
        val_pred_list = []
        test_pred_list = []

        # 每个子模型单独预测
        for local_id in range(args.part_num):
            val_pred = model.single_prediction(valid_data[:, 0], valid_data[:, 1], local_id)
            test_pred = model.single_prediction(test_data[:, 0], test_data[:, 1], local_id)
            val_pred_list.append(val_pred.reshape(-1, 1))
            test_pred_list.append(test_pred.reshape(-1, 1))

        # 拼接所有子模型输出
        val_all_preds = np.hstack(val_pred_list)
        test_all_preds = np.hstack(test_pred_list)

        # SISA 聚合：平均
        val_final = np.mean(val_all_preds, axis=1)
        test_final = np.mean(test_all_preds, axis=1)

        # OR：取最大
        val_or = np.max(val_all_preds, axis=1)
        test_or = np.max(test_all_preds, axis=1)

        # AND：取最小
        val_and = np.min(val_all_preds, axis=1)
        test_and = np.min(test_all_preds, axis=1)

    # 计算指标
    valid_auc_sisa = roc_auc_score(valid_data[:, 2], val_final)
    valid_auc_or_sisa = roc_auc_score(valid_data[:, 2], val_or)
    valid_auc_and_sisa = roc_auc_score(valid_data[:, 2], val_and)

    test_auc_sisa = roc_auc_score(test_data[:, 2], test_final)
    test_auc_or_sisa = roc_auc_score(test_data[:, 2], test_or)
    test_auc_and_sisa = roc_auc_score(test_data[:, 2], test_and)

    # ===================== 输出结果 =====================
    print("\n" + "="*60)
    print("✅ 标准 SISA 聚合结果（LightGCN）")
    print("="*60)
    print(f"Valid AUC        = {valid_auc_sisa:.6f}")
    print(f"Valid AUC OR     = {valid_auc_or_sisa:.6f}")
    print(f"Valid AUC AND    = {valid_auc_and_sisa:.6f}")
    print(f"Test AUC         = {test_auc_sisa:.6f}")
    print(f"Test AUC OR      = {test_auc_or_sisa:.6f}")
    print(f"Test AUC AND     = {test_auc_and_sisa:.6f}")
    print("="*60)

    # 时间统计
    print("\nTraining Time Summary:")
    for i in range(args.part_num):
        print(f"local{i + 1}: {total_time[i]:.6f} seconds")
    local_sum = sum(total_time)
    print(f"total: {local_sum:.6f} seconds")


if __name__ == '__main__':
    begin = perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = {
        'lr': 1e-4,
        'embed_size': 32,
        'batch_size': 2048,
        'data_type': 'retraining',
        'init_std': 1e-4,
        'dataset': 'Mooccube',
        'attack': '0.01',
        'seed': 1024,
        'part_type': 3,
    }
    main(config)
    end = perf_counter()
    print("ALL Time taken:", (end - begin), file=sys.stderr)