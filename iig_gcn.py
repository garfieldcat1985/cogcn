import numpy as np
import scipy as sp
import tensorflow as tf

from utility.batch_test import *

tf.random.set_random_seed(1)
np.random.seed(10)


class iig_gcn(object):

    def __init__(self, data_config, pretrain_data, node_type_1, node_type_2, embeddings):
        # argument settings
        args = global_var.args
        self.adj_type = args.adj_type
        self.pretrain_data = pretrain_data
        self.n_node_type_1 = data_config['n_' + node_type_1]
        self.n_node_type_2 = data_config['n_' + node_type_2]

        self.train_items = data_config['train_items']
        self.weight_size = [64, 64, 64, 64]

        self.n_layers = len(self.weight_size)
        self.n_fold = 1
        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = data_config['batch_size']
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        self.all_u_list = data_config['all_u_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']
        self.all_r_list = data_config['all_r_list']

        self.A_in = data_config['norm_adj']
        self.A_values = tf.placeholder(tf.float32, shape=[len(self.all_v_list)])
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.node1s = tf.placeholder(tf.int32, shape=(None,))
        self.pos_node2s = tf.placeholder(tf.int32, shape=(None,))
        self.neg_node2s = tf.placeholder(tf.int32, shape=(None,))

        self.h0 = tf.placeholder(tf.int32, shape=[None], name='h0')
        self.r0 = tf.placeholder(tf.int32, shape=[None], name='r0')
        self.pos_t0 = tf.placeholder(tf.int32, shape=[None], name='pos_t0')

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=None)

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.iig_embeddings = []
        self.weights = dict()
        self.weights['node_type_1_embedding'] = embeddings
        self.weights['node_type_2_embedding'] = embeddings

        self.n1a_embeddings, self.n2a_embeddings = self._create_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.n1_g_embeddings = tf.nn.embedding_lookup(self.n1a_embeddings, self.node1s)
        self.pos_n2_g_embeddings = tf.nn.embedding_lookup(self.n2a_embeddings, self.pos_node2s)
        self.neg_n2_g_embeddings = tf.nn.embedding_lookup(self.n2a_embeddings, self.neg_node2s)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.n1_g_embeddings, self.pos_n2_g_embeddings, transpose_a=False,
                                       transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.n1_g_embeddings,
                                                                          self.pos_n2_g_embeddings,
                                                                          self.neg_n2_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_node_type_1 + self.n_node_type_2) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_node_type_1 + self.n_node_type_2
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_embed(self):
        # Generate a set of adjacency sub-matrix.
        A = self.A_in
        A_fold_hat = self._split_A_hat(A)

        ego_embeddings = tf.concat([self.weights['node_type_1_embedding'], self.weights['node_type_2_embedding']],
                                   axis=0)
        self.iig_embeddings.append(self.weights['node_type_1_embedding'])
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = side_embeddings

            # non-linear activation.
            ego_embeddings = sum_embeddings

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)
            s0, s1 = tf.split(norm_embeddings, [self.n_node_type_1, self.n_node_type_2], 0)
            self.iig_embeddings.append(s0)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.reduce_sum(all_embeddings, 0)
        n1_g_embeddings, n2_g_embeddings = tf.split(all_embeddings, [self.n_node_type_1, self.n_node_type_2], 0)

        return n1_g_embeddings, n2_g_embeddings

    def create_bpr_loss(self, n1s, pos_n2s, neg_n2s):
        pos_scores = tf.reduce_sum(tf.multiply(n1s, pos_n2s), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(n1s, neg_n2s), axis=1)
        regularizer = tf.nn.l2_loss(n1s) + tf.nn.l2_loss(pos_n2s) + tf.nn.l2_loss(neg_n2s)
        regularizer = regularizer / self.batch_size

        mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer
        reg_loss = tf.constant(0.0, tf.float32, [1])
        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
