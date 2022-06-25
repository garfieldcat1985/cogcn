import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import collections
from utility.parser import parse_args

rd.seed(10)
np.random.seed(10)
args = parse_args()


class CoData(object):
    n_side_info_entity = {}

    def __init__(self, path, filename, batch_size, type_1, type_2, type1_count, type2_count):
        self.path = path
        self.type_1 = type_1
        self.type_2 = type_2
        self.tmp_filename = filename
        train_file = path + '/' + filename + '.txt'

        # get number of users and items
        self.n_side_info_entity[type_1] = type1_count
        self.n_side_info_entity[type_2] = type2_count

        self.exist_entity1s = []
        self.exist_entity2s = []

        self.n_train = 0
        self.neg_pools = {}
        self.n_entity1s = type1_count
        self.n_entity2s = type2_count
        # read train_file

        # n_train: iteractions of all train nodes
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    entity_1_id = int(l[0])
                    self.exist_entity1s.append(entity_1_id)
                    self.n_train += len(items)

        # n_test: interactions of all test nodes

        # self.print_statistics()

        self.R = sp.dok_matrix((self.n_entity1s, self.n_entity2s), dtype=np.float32)

        self.train_entity2s = {}  # train and test data for items(per user)
        self.batch_size = batch_size

        with open(train_file) as f_train:
            for l in f_train.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]
                for i in train_items:
                    self.R[uid, i] = 1.
                self.train_entity2s[uid] = train_items

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(
                self.path + '/' + self.type_1 + '_' + self.type_2 + self.tmp_filename + '_s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(
                self.path + '/' + self.type_1 + '_' + self.type_2 + self.tmp_filename + '_s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(
                self.path + '/' + self.type_1 + '_' + self.type_2 + self.tmp_filename + '_s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/' + self.type_1 + '_' + self.type_2 + self.tmp_filename + '_s_adj_mat.npz',
                        adj_mat)
            sp.save_npz(self.path + '/' + self.type_1 + '_' + self.type_2 + self.tmp_filename + '_s_norm_adj_mat.npz',
                        norm_adj_mat)
            sp.save_npz(self.path + '/' + self.type_1 + '_' + self.type_2 + self.tmp_filename + '_s_mean_adj_mat.npz',
                        mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    # create three adjacency matrix
    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_entity1s + self.n_entity2s, self.n_entity1s + self.n_entity2s),
                                dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_entity1s, self.n_entity1s:self.n_entity1s + self.n_entity2s] = R
        adj_mat[self.n_entity1s:self.n_entity1s + self.n_entity2s, :self.n_entity1s] = R.T

        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()


        def adj_test(adj_mat):
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()

            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            return norm_adj.tocoo()





        # if args.adj_type == 'pre':
        #     norm_adj_mat = adj_test(adj_mat)
        # elif args.adj_type == 'norm':
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))

        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    # Random 100 items per user as negative pools
    def negative_pool(self):
        t1 = time()
        for u in self.train_entity2s.keys():
            neg_items = list(set(range(self.n_entity2s)) - set(self.train_entity2s[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    # sample training data
    def sample(self):
        if self.batch_size <= self.n_entity1s:
            entitiy1s = rd.sample(self.exist_entity1s, self.batch_size)
        else:
            entitiy1s = [rd.choice(self.exist_entity1s) for _ in range(self.batch_size)]

        # sample num positive items for user u
        def sample_pos_items_for_u(u, num):
            pos_items = self.train_entity2s[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_entity2s, size=1)[0]
                if neg_id not in self.train_entity2s[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in entitiy1s:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        return entitiy1s, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def _get_all_data(self, norm_adj):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        lap = norm_adj.tocoo()
        all_h_list = list(lap.row)
        all_t_list = list(lap.col)
        all_v_list = list(lap.data)
        all_r_list = [0] * len(all_h_list)

        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[], [], []]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)
            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)
            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])

        all_h_list, all_t_list, all_r_list, all_v_list = new_h_list, new_t_list, new_r_list, new_v_list

        for i in range(len(all_h_list)):

            if all_h_list[i] == all_t_list[i]:
                all_r_list[i] = 4
            else:
                if all_h_list[i] < self.n_entity1s:
                    all_r_list[i] = 0
                if all_h_list[i] < self.n_entity1s + self.n_entity2s and all_h_list[i] >= self.n_entity1s and \
                        all_t_list[
                            i] < self.n_entity1s:
                    all_r_list[i] = 1

        return all_h_list, all_t_list, all_r_list, all_v_list

    def print_statistics(self):

        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))
