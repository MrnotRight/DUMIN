import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from rnn import dynamic_rnn
from utils import *


class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False):
        with tf.name_scope('Inputs'):
            self.EMBEDDING_DIM = EMBEDDING_DIM
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')
            self.item_user_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='item_user_his_batch_ph')
            self.item_user_his_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name= 'item_user_his_mid_batch_ph')
            self.item_user_his_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name= 'item_user_his_cat_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='his_mask')
            self.item_user_his_mask = tf.placeholder(tf.float32, [None, None], name='item_user_mask')
            self.item_user_his_mid_mask = tf.placeholder(tf.float32, [None, None, None], name='item_user_his_mask')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])
            self.istrain = tf.placeholder(tf.bool, []) #control batchnorm
            self.use_negsampling = use_negsampling
            if self.use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_mid_batch_ph')  # generate 3 item IDs from negative sampling.
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')


        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)
            self.item_user_his_uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.item_user_his_batch_ph)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
            self.item_user_his_mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.item_user_his_mid_batch_ph)

            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.noclk_mid_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
            self.item_user_his_cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.item_user_his_cat_batch_ph)

            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.noclk_cat_batch_ph)

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_user_his_eb = tf.concat([self.item_user_his_mid_batch_embedded, self.item_user_his_cat_batch_embedded], -1)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb*tf.expand_dims(self.mask,-1), 1)

        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat([self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]], -1)
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,[-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], 36])
            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1) # [B, T, 3, 2*H]
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2) # [B, T, 2*H] 未点击的三个item的emb和
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1) #[B, 2*H] T个未点击的三个item的emb的和的和


    def build_fcn_net(self, inp):
        dnn1 = tf.layers.dense(inp, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'prelu1')
        dnn2 = tf.layers.dense(dnn1, 100, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 40, activation=None, name='f3')
        dnn3 = prelu(dnn2, 'prelu3')
        dnn4 = tf.layers.dense(dnn3, 2, activation=None, name='f4')
        self.y_hat = tf.nn.softmax(dnn4) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat + 1e-6) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
               self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag = None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_+ 1e-6), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_ + 1e-6), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        #bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        #dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(in_, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def train(self, sess, inps):
        if self.use_negsampling:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.item_user_his_batch_ph:inps[6],
                self.item_user_his_mask:inps[7],
                self.item_user_his_mid_batch_ph:inps[8],
                self.item_user_his_cat_batch_ph:inps[9],
                self.item_user_his_mid_mask:inps[10],
                self.target_ph: inps[11],
                self.seq_len_ph: inps[12],
                self.lr: inps[13],
                self.noclk_mid_batch_ph: inps[14],
                self.noclk_cat_batch_ph: inps[15],
                self.istrain:True
            })
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.item_user_his_batch_ph: inps[6],
                self.item_user_his_mask: inps[7],
                self.item_user_his_mid_batch_ph: inps[8],
                self.item_user_his_cat_batch_ph: inps[9],
                self.item_user_his_mid_mask: inps[10],
                self.target_ph: inps[11],
                self.seq_len_ph: inps[12],
                self.lr: inps[13],
                self.istrain:True
            })
        return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.item_user_his_batch_ph: inps[6],
                self.item_user_his_mask: inps[7],
                self.item_user_his_mid_batch_ph: inps[8],
                self.item_user_his_cat_batch_ph: inps[9],
                self.item_user_his_mid_mask: inps[10],
                self.target_ph: inps[11],
                self.seq_len_ph: inps[12],
                self.noclk_mid_batch_ph: inps[13],
                self.noclk_cat_batch_ph: inps[14],
                self.istrain:False
            })
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.item_user_his_batch_ph: inps[6],
                self.item_user_his_mask: inps[7],
                self.item_user_his_mid_batch_ph: inps[8],
                self.item_user_his_cat_batch_ph: inps[9],
                self.item_user_his_mid_mask: inps[10],
                self.target_ph: inps[11],
                self.seq_len_ph: inps[12],
                self.istrain:False
            })
        return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


class Model_WideDeep(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_WideDeep, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        # Fully connected layer
        #bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        #dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = tf.layers.dense(inp, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'p1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'p2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        d_layer_wide = tf.concat([tf.concat([self.item_eb,self.item_his_eb_sum], axis=-1),
                                self.item_eb * self.item_his_eb_sum], axis=-1)
        d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')
        self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()


class Model_DNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                                          ATTENTION_SIZE)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp)

class Model_PNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_PNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum], 1)

        # Fully connected layer
        self.build_fcn_net(inp)


class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                           ATTENTION_SIZE)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea], -1)
        # Fully connected layer
        self.build_fcn_net(inp)


class DIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)

        # RNN layer(-s) interest extractor layer base GRU
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,  # [B, T, 36]
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,  # [B, 1]
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)
        if self.use_negsampling:
            aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.noclk_item_his_eb[:, 1:, :],
                                             self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),  # [B, T, 1]
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
             final_state2], 1)
        self.build_fcn_net(inp)


class Model_GRU4REC(Model):
    def __init__(self,  n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_GRU4REC, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)

        with tf.name_scope('rnn1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            _, final_state_1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                           sequence_length=self.seq_len_ph, dtype=tf.float32,
                                           scope="gru2")
        with tf.name_scope('rnn1'):
            item_user_len = tf.reduce_sum(self.item_user_his_mask,axis=-1)
            item_user_rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_user_his_uid_batch_embedded,
                                                   sequence_length=item_user_len, dtype=tf.float32,
                                                   scope="gru3")
            _, final_state_2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=item_user_rnn_outputs,
                                           sequence_length=item_user_len, dtype=tf.float32,
                                           scope="gru4")

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state_1,final_state_2], 1)
        # Fully connected layer
        self.build_fcn_net(inp)

class Model_SVDPP(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_SVDPP, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        self.uid_b = tf.get_variable("uid_b", [n_uid, 1])
        self.user_b = tf.nn.embedding_lookup(self.uid_b, self.uid_batch_ph)
        self.mid_b = tf.get_variable("mid_b", [n_mid, 1])
        self.item_b = tf.nn.embedding_lookup(self.mid_b, self.mid_batch_ph)
        # print(self.item_b)
        self.mu = tf.get_variable('mu', [], initializer=tf.truncated_normal_initializer)
        self.user_w = tf.get_variable('user_w', [EMBEDDING_DIM * 3, EMBEDDING_DIM * 2],initializer=tf.truncated_normal_initializer)
        neighbors_rep_seq = tf.concat([self.item_user_his_uid_batch_embedded,tf.reduce_sum(self.item_user_his_eb, axis=2)],axis=-1)
        user_rep = tf.concat([self.uid_batch_embedded,self.item_his_eb_sum],axis=-1)
        user_rep = tf.matmul(user_rep,self.user_w)
        print(user_rep)
        neighbors_norm = tf.sqrt(tf.expand_dims(tf.norm(neighbors_rep_seq, 1, (1, 2)),1))
        neighbors_norm = tf.where(neighbors_norm>0,neighbors_norm,tf.ones_like(neighbors_norm))
        neighbor_emb = tf.reduce_sum(neighbors_rep_seq,1)/neighbors_norm
        neighbor_emb = tf.matmul(neighbor_emb, self.user_w)
        print(neighbor_emb)
        score = tf.reduce_sum(self.item_eb * (user_rep+neighbor_emb),1)+tf.reshape(self.user_b,[-1])+tf.reshape(self.item_b, [-1])+self.mu
        pred = tf.reshape(tf.nn.sigmoid(score), [-1, 1])
        self.y_hat = tf.concat([pred,1-pred], -1)+0.00000001
        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()


class Model_DUMN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DUMN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,ATTENTION_SIZE)

        with tf.name_scope('DUMN'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
            user_feat = tf.concat([self.uid_batch_embedded,att_fea],axis=-1)
            print(self.item_user_his_eb)
            item_user_his_attention_output = din_attention(tf.tile(self.item_eb, [1, tf.shape(self.item_user_his_eb)[1]*tf.shape(self.item_user_his_eb)[2]]),
                                                           tf.reshape(self.item_user_his_eb,[-1, tf.shape(self.item_user_his_eb)[2],36]),
                                                           ATTENTION_SIZE, tf.reshape(self.item_user_his_mid_mask,[-1, tf.shape(self.item_user_his_mid_mask)[2]]),
                                                           need_tile=False)
            item_user_his_att = tf.reshape(tf.reduce_sum(item_user_his_attention_output, 1),[-1,tf.shape(self.item_user_his_eb)[1],36])
            item_user_bhvs_feat = tf.concat([self.item_user_his_uid_batch_embedded,item_user_his_att],axis=-1)
            sim_score = user_similarity(user_feat,item_user_bhvs_feat,need_tile=True)*self.item_user_his_mask
            sim_score_sum = tf.reduce_sum(sim_score,axis=-1,keep_dims=True)
            sim_att = tf.reduce_sum(item_user_bhvs_feat*tf.expand_dims(sim_score, -1),axis=1)
            
        inp = tf.concat([user_feat, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,sim_att,sim_score_sum], -1)
        # Fully connected layer
        self.build_fcn_net(inp)
        # for tf_var in tf.trainable_variables():
        #     print(tf_var)

class Model_DUMIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DUMIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # din evolution user_sim
        with tf.name_scope("DIN"):
            attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1) #[B,2*H]
            tf.summary.histogram('att_fea', att_fea)
        with tf.name_scope('evolution'):
            rnn_outputs, evolu_interest = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,  # [B, T, 36]
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,  # [B, 1]
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

            aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], \
                                                self.item_his_eb[:, 1:, :], \
                                                self.noclk_item_his_eb[:, 1:, :],\
                                                self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1
        self_interest = tf.concat([att_fea, evolu_interest], axis=-1)
        with tf.name_scope('neibors_interest'):
            user_sim_interest = user_similar_interst(self_interest, self.item_user_his_uid_batch_embedded,\
                                                                   self.item_user_his_eb, self.item_user_his_mask, \
                                                                   self.item_user_his_mid_mask)
            user_sim_interest = tf.reduce_sum(user_sim_interest, 1)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self_interest,  user_sim_interest], -1)
        self.build_fcn_net(inp)


