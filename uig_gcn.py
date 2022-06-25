import numpy
import numpy as np
import scipy as sp
import tensorflow as tf


from utility.batch_test import *

tf.random.set_random_seed(1)
np.random.seed(10)


class uig_gcn(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        args = global_var.args
        self.adj_type = args.adj_type
        self.pretrain_data = pretrain_data

        self.half_layer_num = 2
        self.l2_reg = 0.0001

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']


        self.train_items = data_config['train_items']
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.n_fold = 2
        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.embde_dim = args.embed_size
        self.batch_size = data_config['batch_size']
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        self.all_u_list = data_config['all_u_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']

        self.A_in = data_config['norm_adj']
        self.A_values = tf.placeholder(tf.float32, shape=[len(self.all_v_list)])
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.h0 = tf.placeholder(tf.int32, shape=[None], name='h0')
        self.r0 = tf.placeholder(tf.int32, shape=[None], name='r0')
        self.pos_t0 = tf.placeholder(tf.int32, shape=[None], name='pos_t0')

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = dict()
        self.ref_weights = dict()
        self.weights['user_embedding'] = global_var.global_all_weights['user_embedding']
        self.weights['item_embedding'] = global_var.global_all_weights['item_embedding']
        self.ref_weights["item_embedding_co_buy"] = global_var.iig_embedding_buys
        self.ref_weights["item_embedding_co_view"] = global_var.iig_embedding_views

        self.ua_embeddings, self.ia_embeddings = self._create_embed()
        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False,
                                       transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_embed(self):
        # Generate a set of adjacency sub-matrix.
        A = self.A_in
        A_fold_hat = self._split_A_hat(A)

        ref_item_embedding_co_buy = self.ref_weights['item_embedding_co_buy'][0]
        ref_item_embedding_co_view = self.ref_weights['item_embedding_co_view'][0]

        v1 = tf.expand_dims(self.weights['item_embedding'], 2)
        v2 = tf.expand_dims(ref_item_embedding_co_buy, 2)
        v3 = tf.expand_dims(ref_item_embedding_co_view, 2)

        sm0 = tf.concat([v1, v2, v3], 2)
        sm0 = tf.nn.softmax(sm0, dim=2)
        item_embedding_total = sm0[:, :, 0] * self.weights['item_embedding'] + sm0[:, :, 1] * ref_item_embedding_co_buy \
                               + sm0[:, :, 2] * ref_item_embedding_co_view

        ego_embeddings = tf.concat([self.weights['user_embedding'], item_embedding_total], axis=0)

        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = side_embeddings

            ego_embeddings = sum_embeddings

            # message dropout.
            # ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout)

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            tmp0, tmp1 = tf.split(norm_embeddings, [self.n_users, self.n_items], 0)
            if k + 1 <= len(self.ref_weights['item_embedding_co_buy']) - 1:
                ref_item_embedding_co_buy = self.ref_weights['item_embedding_co_buy'][k + 1]
                ref_item_embedding_co_view = self.ref_weights['item_embedding_co_view'][k + 1]

                v1 = tf.expand_dims(tmp1, 2)
                v2 = tf.expand_dims(ref_item_embedding_co_buy, 2)
                v3 = tf.expand_dims(ref_item_embedding_co_view, 2)
                sm0 = tf.concat([v1, v2, v3], 2)
                sm0 = tf.nn.softmax(sm0, dim=2)
                s1 = sm0[:, :, 0] * tmp1 + sm0[:, :, 1] * ref_item_embedding_co_buy + sm0[:, :,
                                                                                      2] * ref_item_embedding_co_view
                norm_embeddings = tf.concat([tmp0, s1], axis=0)



            # else:
            #     ref_item_embedding_co_buy = self.ref_weights['item_embedding_co_buy'][
            #         len(self.ref_weights['item_embedding_co_buy']) - 1]
            #     ref_item_embedding_co_view = self.ref_weights['item_embedding_co_view'][
            #         len(self.ref_weights['item_embedding_co_view']) - 1]



            all_embeddings += [norm_embeddings]

        # all_embeddings = tf.concat(all_embeddings, 1)
        all_embeddings = tf.reduce_sum(all_embeddings, 0)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)



        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer / self.batch_size

        mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer
        reg_loss = tf.constant(0.0, tf.float32, [1])
        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
