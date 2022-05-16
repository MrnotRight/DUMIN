import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
#from tensorflow.python.ops.rnn_cell_impl import  _Linear
from tensorflow.contrib.rnn.python.ops.rnn_cell import _Linear
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18
NUM_HEAD = 6
best_auc = 0.0


class VecAttGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(VecAttGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units
  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)
  def call(self, inputs, state, att_score=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _Linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([inputs, r_state]))
    u = (1.0 - att_score) * u
    new_h = u * state + (1 - u) * c
    return new_h, new_h

def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc


def user_similarity(user_feat,item_user_bhvs,need_tile=True):
    user_feats = tf.tile(user_feat, [1, tf.shape(item_user_bhvs)[1]]) if need_tile else user_feat
    user_feats = tf.reshape(user_feats, tf.shape(item_user_bhvs))
    pooled_len_1 = tf.sqrt(tf.reduce_sum(user_feats * user_feats, -1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(item_user_bhvs * item_user_bhvs, -1))
    pooled_mul_12 = tf.reduce_sum(user_feats * item_user_bhvs, -1)
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
    return score



def din_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False,need_tile=True):
    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    print(facts)
    queries = tf.tile(query, [1, tf.shape(facts)[1]]) if need_tile else query
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    # print(din_all)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag,reuse=tf.AUTO_REUSE)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag,reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag,reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]
    else:
        scores = tf.nn.sigmoid(scores)

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
    #     scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])#[B,T]
    #     output = facts * tf.expand_dims(scores, -1)
    #     output = tf.reshape(output, tf.shape(facts))#[B,T,H]
        scores = tf.transpose(scores, [0,2,1]) #[B,T,1]
        output = tf.multiply(facts, scores) #[B,T,H]
    return output


def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output


def user_similar_interst(query_interest, item_users_keys, item_users_hist, item_user_mask, item_user_hist_mask):
    # query_interest [B,X]     item_users_keys:[B,U,E]    item_users_hist_keys:[B,U,T,E]    item_user_hist_mask:[B,U,T],item_user_mask:[B, U]
    query_trans = tf.layers.dense(query_interest, 3*NUM_HEAD*HIDDEN_SIZE, name='transquery')
    query, self_key, self_value = tf.split(query_trans, axis=1, num_or_size_splits=3) #[B,n*H]
    keys = tf.layers.dense(item_users_hist, NUM_HEAD*HIDDEN_SIZE, name='transkeys') #[B,U,T,n*H]
    values = tf.layers.dense(item_users_hist, NUM_HEAD*HIDDEN_SIZE, name='transvalues') #[B,U,T,n*H]
    query = tf.expand_dims(query, 1) #[B,1,n*H]
    query = tf.tile(query, [1, tf.shape(keys)[1],1]) #[B,U,n*H]
    query = tf.reshape(query, [-1, tf.shape(keys)[1], NUM_HEAD, HIDDEN_SIZE]) #[B,U,N,H]
    query = tf.expand_dims(query, -2) #[B,U,N,1,H]

    self_key = tf.expand_dims(self_key, 1) #[B,1,N*H]
    self_key = tf.tile(self_key, [1, tf.shape(keys)[1], 1])  # [B,U,N*H]
    self_key = tf.expand_dims(self_key, 2)  # [B,U,1,N*H]
    keys = tf.concat([keys, self_key], axis=-2) #[B,U,T+1,N*H]

    keys = tf.reshape(keys, [tf.shape(keys)[0], tf.shape(keys)[1], tf.shape(keys)[2], NUM_HEAD, HIDDEN_SIZE]) #[B,U,T+1,N,H]
    keys = tf.transpose(keys, [0,1,3,2,4]) #[B,U,N,T+1,H]
    attention_scores = tf.matmul(query, tf.transpose(keys, [0,1,2,4,3])) #[B,U,N,1,T+1]
    attention_scores = tf.div(attention_scores, tf.sqrt(float(HIDDEN_SIZE)))

    # mask
    padding_mask = tf.ones([tf.shape(item_user_hist_mask)[0], tf.shape(item_user_hist_mask)[1], 1], dtype=tf.float32)
    mask = tf.concat([item_user_hist_mask, padding_mask], axis=-1) #[B,U,T+1]
    mask = tf.expand_dims(mask, 2)
    mask = tf.expand_dims(mask, 2) #[B,U,1,1,T+1]
    mask = tf.tile(mask, [1,1,NUM_HEAD, 1, 1]) #[B,U,N,1,T+1]
    paddings = tf.ones_like(attention_scores) * (-2 ** 32 + 1)
    mask = tf.equal(mask, tf.ones_like(mask))
    attention_scores = tf.where(mask, attention_scores, paddings)
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)  # [B,U,N,1,T+1]

    self_value = tf.expand_dims(self_value, 1) #[B,1,N*H]
    self_value = tf.tile(self_value, [1, tf.shape(values)[1], 1]) #[B,U,N*H]
    self_value = tf.expand_dims(self_value, 2) #[B,U,1,N*H]
    values = tf.concat([values, self_value], axis=-2)  # [B,U,T+1,N*H]

    values = tf.reshape(values, [tf.shape(values)[0], tf.shape(values)[1], tf.shape(values)[2], NUM_HEAD, HIDDEN_SIZE])  # [B,U,T+1,N,H]
    values = tf.transpose(values, [0, 1, 3, 2, 4])  # [B,U,N,T+1,H]
    attention_output = tf.matmul(attention_scores, values)  # [B,U,N,1,H]
    attention_output = tf.reshape(attention_output,[tf.shape(attention_output)[0], tf.shape(attention_output)[1], NUM_HEAD*HIDDEN_SIZE])  # [B,U,N*H]

    # pooling
    # attention_pool_output, _ = torch.max(attention_output, dim=1) #[B,N*H]
    query_interest = tf.expand_dims(query_interest, 1)  # [B,1,X]
    query_interest = tf.tile(query_interest, [1, tf.shape(attention_output)[1], 1])  # [B,U,X]
    #pooling_input = tf.concat([attention_output, query_interest], axis=-1) #[B,U,X+H]
    pooling_input = tf.concat([attention_output, item_users_keys, query_interest], axis=-1) #[B,U,X+H]
    #pooling_input = tf.reshape(pooling_input, [-1,tf.shape(pooling_input)[1], 2*EMBEDDING_DIM+NUM_HEAD*HIDDEN_SIZE])
    pooling_output = tf.layers.dense(pooling_input, 100, activation=tf.nn.sigmoid, name='user_pool1')
    pooling_output = tf.layers.dense(pooling_output, 50, activation=tf.nn.sigmoid, name='user_pool2')
    pooling_output = tf.layers.dense(pooling_output, 1, activation=None, name='user_pool3') #[B,U,1]
    pooling_output = tf.reduce_sum(pooling_output, -1)
    pooling_scores = tf.expand_dims(pooling_output, 1)  #[B,1,U]

    user_mask = tf.expand_dims(item_user_mask, 1) #[B,1,U]
    paddings = tf.ones_like(user_mask) * (-2**32 + 1)
    user_mask = tf.equal(user_mask, tf.ones_like(user_mask))
    pooling_scores = tf.where(user_mask, pooling_scores, paddings)
    pooling_scores = tf.nn.softmax(pooling_scores, axis=-1)  # [B,1,U]
    # user_embdding_similarity
    attention_pool_output = tf.matmul(pooling_scores, tf.concat([attention_output, item_users_keys], axis=-1)) #[B,1,N*H]
    #pooling_scores = tf.reshape(pooling_scores, [-1, 1, tf.shape(item_users_hist)[1]])
    #attention_pool_user = tf.matmul(pooling_scores, item_users_keys) #[B,1,E]
    #return attention_pool_output, attention_pool_user
    return attention_pool_output
