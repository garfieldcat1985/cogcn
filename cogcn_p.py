import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from time import time
import numpy
import tensorflow as tf
from iig_gcn import iig_gcn
from utility.batch_test import test

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.random.set_random_seed(1)
numpy.random.seed(10)

import global_var
from uig_gcn import uig_gcn
from utility.load_data import Data
cpu_num = multiprocessing.cpu_count()
from utility.load_co_data import CoData
from utility.parser import parse_args


def init_embedding_for_entities():
    global_var.global_all_weights = {}
    initializer = tf.contrib.layers.xavier_initializer()
    global_var.global_all_weights['user_embedding'] = tf.Variable(initializer([user_num, global_var.args.embed_size]),
                                                                  name='user_embedding', dtype=tf.float32)
    global_var.global_all_weights['item_embedding'] = tf.Variable(initializer([item_num, global_var.args.embed_size]),
                                                                  name='item_embedding', dtype=tf.float32)

    global_var.global_all_weights['item_embedding_in_co_buy'] = tf.Variable(
        initializer([item_num, global_var.args.embed_size]),
        name='item_embedding_in_co_buy', dtype=tf.float32)
    global_var.global_all_weights['item_embedding_in_co_view'] = tf.Variable(
        initializer([item_num, global_var.args.embed_size]),
        name='item_embedding_in_co_view', dtype=tf.float32)


if __name__ == '__main__':
    global_var.args = parse_args()
    global_var.Ks = eval(global_var.args.Ks)

    global_var.data_generator = Data(path=global_var.args.data_path + global_var.args.dataset,
                                     batch_size=global_var.args.batch_size)

    user_num = global_var.data_generator.n_users
    item_num = global_var.data_generator.n_items

    global_var.co_buy_data_generator = CoData('Data/' + global_var.args.dataset, 'item_co_buy_triple',
                                              global_var.args.batch_size, 'item',
                                              'item', item_num, item_num)
    global_var.co_view_data_generator = CoData('Data/' + global_var.args.dataset, 'item_co_view_triple',
                                               global_var.args.batch_size, 'item',
                                               'item', item_num, item_num)

    n_batch = global_var.data_generator.n_train // global_var.args.batch_size + 1

    global_var.USR_NUM, global_var.ITEM_NUM = global_var.data_generator.n_users, global_var.data_generator.n_items
    N_TRAIN, N_TEST = global_var.data_generator.n_train, global_var.data_generator.n_test
    global_var.BATCH_SIZE = global_var.args.batch_size
    global_var.test_executor = ProcessPoolExecutor(max_workers=cpu_num - 1)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    init_embedding_for_entities()
    ####################################################
    config_side = dict()
    config_side['n_item'] = global_var.co_buy_data_generator.n_entity1s
    config_side['train_items'] = global_var.co_buy_data_generator.train_entity2s
    config_side['batch_size'] = global_var.co_buy_data_generator.batch_size
    side_plain_adj, side_norm_adj, side_mean_adj = global_var.co_buy_data_generator.get_adj_mat()
    config_side['norm_adj'] = side_norm_adj
    all_u_list, all_t_list, all_r_list, all_v_list = global_var.co_buy_data_generator._get_all_data(side_norm_adj)
    config_side['all_u_list'] = all_u_list
    config_side['all_r_list'] = all_r_list
    config_side['all_t_list'] = all_t_list
    config_side['all_v_list'] = all_v_list
    model_co_buy = iig_gcn(config_side, None, 'item', 'item',
                           global_var.global_all_weights['item_embedding_in_co_buy'])

    global_var.iig_embedding_buys = model_co_buy.iig_embeddings

    config_side = dict()
    config_side['n_item'] = global_var.co_view_data_generator.n_entity1s
    config_side['train_items'] = global_var.co_view_data_generator.train_entity2s
    config_side['batch_size'] = global_var.co_view_data_generator.batch_size
    side_plain_adj, side_norm_adj, side_mean_adj = global_var.co_view_data_generator.get_adj_mat()
    config_side['norm_adj'] = side_norm_adj
    all_u_list, all_t_list, all_r_list, all_v_list = global_var.co_view_data_generator._get_all_data(side_norm_adj)
    config_side['all_u_list'] = all_u_list
    config_side['all_r_list'] = all_r_list
    config_side['all_t_list'] = all_t_list
    config_side['all_v_list'] = all_v_list
    model_co_view = iig_gcn(config_side, None, 'item', 'item',
                            global_var.global_all_weights['item_embedding_in_co_view'])
    global_var.iig_embedding_views = model_co_view.iig_embeddings

    config = dict()
    config['n_users'] = global_var.data_generator.n_users
    config['n_items'] = global_var.data_generator.n_items
    config['train_items'] = global_var.data_generator.train_items
    config['batch_size'] = global_var.data_generator.batch_size
    plain_adj, norm_adj, mean_adj = global_var.data_generator.get_adj_mat()
    config['norm_adj'] = norm_adj
    all_u_list, all_t_list, all_r_list, all_v_list = global_var.data_generator._get_all_data(norm_adj)
    config['all_u_list'] = all_u_list
    config['all_r_list'] = all_r_list
    config['all_t_list'] = all_t_list
    config['all_v_list'] = all_v_list
    model = uig_gcn(config, None)

    ##########################################
    total_loss = None
    total_loss = model.loss + model_co_buy.loss + model_co_view.loss
    total_opt = tf.train.AdamOptimizer(learning_rate=global_var.args.lr).minimize(total_loss)

    sess.run(tf.global_variables_initializer())
    ######################################################
    for epoch in range(1000):
        t1 = time()
        ###################################
        loss_value = 0
        for idx in range(n_batch):
            co_buy_node1s, co_buy_pos_node2s, co_buy_neg_node2s = global_var.co_buy_data_generator.sample()
            co_view_node1s, co_view_pos_node2s, co_view_neg_node2s = global_var.co_view_data_generator.sample()

            users, pos_items, neg_items = global_var.data_generator.sample()
            _, batch_loss = sess.run(
                [total_opt, total_loss],
                feed_dict={model_co_buy.node1s: co_buy_node1s, model_co_buy.pos_node2s: co_buy_pos_node2s,
                           model_co_buy.neg_node2s: co_buy_neg_node2s,

                           model_co_view.node1s: co_view_node1s, model_co_view.pos_node2s: co_view_pos_node2s,
                           model_co_view.neg_node2s: co_view_neg_node2s,

                           model.users: users, model.pos_items: pos_items,
                           model.neg_items: neg_items
                           })
            loss_value = loss_value + batch_loss

        ####################################

        if (epoch + 1) % 10 != 0:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f]' % (
                epoch, time() - t1, loss_value)
            print(perf_str)

            continue

        t2 = time()
        users_to_test = list(global_var.data_generator.test_set.keys())
        ret = test(sess, model, users_to_test)
        t3 = time()

        loss_loger.append(loss_value)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if global_var.args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss_value, ret['recall'][0], ret['recall'][1],
                        ret['precision'][0], ret['precision'][1], ret['hit_ratio'][0], ret['hit_ratio'][1],
                        ret['ndcg'][0], ret['ndcg'][1])
            print(perf_str)
