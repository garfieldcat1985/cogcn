'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import multiprocessing
from functools import partial

import numpy as np


import global_var
import utility.metrics as metrics
import heapq
np.random.seed(10)
cpu_num = multiprocessing.cpu_count()


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = x[2]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = x[3]
    item_num = x[4]
    ks = x[5]
    test_flag = x[6]
    all_items = set(range(item_num))

    test_items = list(all_items - set(training_items))

    if test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, ks)

    return get_performance(user_pos_test, r, auc, ks)


# def set_global():
#     global data_generator_1
#     data_generator_1 = global_var.data_generator
#     global USR_NUM_1
#     USR_NUM_1 = global_var.USR_NUM
#     global ITEM_NUM_1
#     ITEM_NUM_1 = global_var.ITEM_NUM
#     global args_1
#     args_1=global_var.args
#     global Ks_1
#     Ks_1=global_var.Ks
#     global BATCH_SIZE_1
#     BATCH_SIZE_1=global_var.BATCH_SIZE


def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(global_var.Ks)), 'recall': np.zeros(len(global_var.Ks)),
              'ndcg': np.zeros(len(global_var.Ks)),
              'hit_ratio': np.zeros(len(global_var.Ks)), 'auc': 0.}



    # 打开进程/线程池

    p_func = partial(test_one_user)
    u_batch_size = global_var.BATCH_SIZE * 2
    i_batch_size = global_var.BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    result_save = []
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:

            n_item_batchs = global_var.ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), global_var.ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, global_var.ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                  model.pos_items: item_batch})
                else:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                  model.pos_items: item_batch,
                                                                  model.node_dropout: [0.] * len(
                                                                      eval(global_var.args.layer_size)),
                                                                  model.mess_dropout: [0.] * len(
                                                                      eval(global_var.args.layer_size))})
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == global_var.ITEM_NUM
        else:
            item_batch = range(global_var.ITEM_NUM)

            if drop_flag == False:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                            model.pos_items: item_batch})
            else:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                            model.pos_items: item_batch,
                                                            model.node_dropout: [0.],
                                                            model.mess_dropout: [0.]})

        training_items = [global_var.data_generator.train_items[e] for e in user_batch]
        user_pos_tests = [global_var.data_generator.test_set[e] for e in user_batch]
        item_num = [global_var.ITEM_NUM] * len(user_batch)
        ks = [global_var.Ks] * len(user_batch)
        test_flag = [global_var.args.test_flag] * len(user_batch)

        user_batch_rating_uid = zip(rate_batch, user_batch, training_items, user_pos_tests, item_num, ks, test_flag)
        batch_result_tmp = global_var.test_executor.map(test_one_user, user_batch_rating_uid)

        batch_result_tmp1 = [e for e in batch_result_tmp]
        batch_result = batch_result_tmp1.copy()
        count += len(batch_result)

        for re in batch_result:
            result_save.append([re['ndcg'][0], re['hit_ratio'][0]])
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users


    assert count == n_test_users
    # pool.close()

    # np.savetxt(args.dataset + 'agcf2.txt', np.asarray(result_save))

    return result
