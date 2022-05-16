import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0

def prepare_data(input, target, maxlen = None, user_maxlen=None, use_negsampling=False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input] #用户历史点击物品和物品类别的长度
    seqs_mid = [inp[3] for inp in input] #用户历史点击物品序列
    seqs_cat = [inp[4] for inp in input] #用户历史点击物品类别序列
    lengths_s_user = [len(s[5]) for s in input] #点过物品i的用户数
    seqs_user = [inp[5] for inp in input] #点过物品i的用户列表
    seqs_user_mid = [inp[6] for inp in input] #点过物品i的用户之前的历史点击物品序列
    seqs_user_cat = [inp[7] for inp in input] #点过物品i的用户之前的历史点击物品类别序列
    item_user_mid_length = 0
    if use_negsampling:
        noclk_seqs_mid = [inp[8] for inp in input]
        noclk_seqs_cat = [inp[9] for inp in input]

    if maxlen is not None:
        new_seqs_mid = [] #存储截取最大长度后的用户u点击过的物品序列
        new_seqs_cat = [] #存储截取最大长度后的用户u点击过的物品类别序列
        new_lengths_x = [] #存储截取最大长度用户u历史点击长度
        new_seqs_user_mid = [] #存储截取最大长度后点击过物品i的所有用户的历史点击物品序列
        new_seqs_user_cat = [] #存储截取最大长度后点击过物品i的所有用户的历史点击物品类别序列
        if use_negsampling:
            new_noclk_seqs_mid = []  # 存储截取最大长度后的用户u未点击过的物品类别序列
            new_noclk_seqs_cat = []  # 存储截取最大长度后的用户u未点击过的物品类别序列
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_lengths_x.append(maxlen)
                if use_negsampling:
                    new_noclk_seqs_mid.append(inp[8][l_x - maxlen:])
                    new_noclk_seqs_cat.append(inp[9][l_x - maxlen:])
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_lengths_x.append(l_x)
                if use_negsampling:
                    new_noclk_seqs_mid.append(inp[8])
                    new_noclk_seqs_cat.append(inp[9])
        for inp in input:
            one_sample_user_mid = [] #存储每一条样本中点击过物品i的所有用户的历史点击物品序列
            one_sample_user_cat = [] #存储每一条样本中点击过物品i的所有用户的历史点击物品类别序列
            for user_mid in inp[6]:
                len_user_mid = len(user_mid)
                if len_user_mid>maxlen:
                    item_user_mid_length = maxlen
                    one_sample_user_mid.append(user_mid[len_user_mid-maxlen:])
                else:
                    if len_user_mid>item_user_mid_length:
                        item_user_mid_length = len_user_mid
                    one_sample_user_mid.append(user_mid)
            new_seqs_user_mid.append(one_sample_user_mid)

            for user_cat in inp[7]:
                len_user_cat = len(user_cat)
                if len_user_cat>maxlen:
                    one_sample_user_cat.append(user_cat[len_user_cat-maxlen:])
                else:
                    one_sample_user_cat.append(user_cat)
            new_seqs_user_cat.append(one_sample_user_cat)
        lengths_x = new_lengths_x #存储截取最大长度用户u历史点击长度
        seqs_mid = new_seqs_mid  #存储截取最大长度后的用户u点击过的物品序列
        seqs_cat = new_seqs_cat  #存储截取最大长度后的用户u点击过的物品类别序列
        seqs_user_mid = new_seqs_user_mid #存储截取最大长度后点击过物品i的所有用户的历史点击物品序列
        seqs_user_cat = new_seqs_user_cat #存储截取最大长度后点击过物品i的所有用户的历史点击物品类别序列

        if len(lengths_x) < 1:
            return None, None, None, None

    if user_maxlen is not None:
        new_seqs_user = [] #每条样本中截取最大用户长度后的剩余用户序列
        new_lengths_s_user = [] #每条样本中截取最大用户长度后的剩余用户长度
        new_seqs_user_mid = [] #每条样本中截取最大组用户长度后剩余用户的历史物品点击序列
        new_seqs_user_cat = [] #每条样本中截取最大组用户长度后剩余用户的历史物品类别点击序列
        for l_x, inp in zip(lengths_s_user, input):
            if l_x > user_maxlen:
                new_seqs_user.append(inp[5][l_x - user_maxlen:])
                new_lengths_s_user.append(user_maxlen)
            else:
                new_seqs_user.append(inp[5])
                new_lengths_s_user.append(l_x)
        for one_sample_user_mid in seqs_user_mid:
            len_one_sample_user_mid = len(one_sample_user_mid)
            if len_one_sample_user_mid>user_maxlen:
                new_seqs_user_mid.append(one_sample_user_mid[len_one_sample_user_mid-user_maxlen:])
            else:
                new_seqs_user_mid.append(one_sample_user_mid)

        for one_sample_user_cat in seqs_user_cat:
            len_one_sample_user_cat = len(one_sample_user_cat)
            if len_one_sample_user_cat > user_maxlen:
                new_seqs_user_cat.append(one_sample_user_cat[len_one_sample_user_cat - user_maxlen:])
            else:
                new_seqs_user_cat.append(one_sample_user_cat)
        seqs_user = new_seqs_user
        seqs_user_mid = new_seqs_user_mid
        seqs_user_cat = new_seqs_user_cat


    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x) #用户点击历史长度
    user_maxlen_x = numpy.max(lengths_s_user) #用户组最大长度
    user_maxlen_x = user_maxlen_x if user_maxlen_x > 0 else 1
    item_user_mid_length = item_user_mid_length if item_user_mid_length > 0 else 1
    if use_negsampling:
        neg_samples = len(noclk_seqs_mid[0][0]) #随机负采样个数
    # print(maxlen_x,user_maxlen_x,item_user_mid_length)

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    item_user_his = numpy.zeros((n_samples, user_maxlen_x)).astype('int64')
    item_user_his_mid = numpy.zeros((n_samples, user_maxlen_x,item_user_mid_length)).astype('int64')
    item_user_his_cat = numpy.zeros((n_samples, user_maxlen_x,item_user_mid_length)).astype('int64')
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32') # mask his seq
    item_user_his_mask = numpy.zeros((n_samples, user_maxlen_x)).astype('float32') # mask item_user
    item_user_his_mid_mask = numpy.zeros((n_samples, user_maxlen_x,item_user_mid_length)).astype('float32') # mask every item_user hist
    if use_negsampling:
        noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
        noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    if use_negsampling:
        for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)): #填充历史点击序列
            mid_mask[idx, :lengths_x[idx]] = 1.
            mid_his[idx, :lengths_x[idx]] = s_x
            cat_his[idx, :lengths_x[idx]] = s_y
            noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
            noclk_cat_his[idx, :lengths_x[idx], :] = no_sy
    else:
        for idx, [s_x, s_y] in enumerate(zip(seqs_mid, seqs_cat)): #填充历史点击序列
            mid_mask[idx, :lengths_x[idx]] = 1.
            mid_his[idx, :lengths_x[idx]] = s_x
            cat_his[idx, :lengths_x[idx]] = s_y
    for idx,x in enumerate(seqs_user): #填充组用户序列
        item_user_his_mask[idx, :len(x)] = 1.
        item_user_his[idx, :len(x)] = x
    for idx,x in enumerate(seqs_user_mid):#填充组用户点击序列
        for idy, y in enumerate(x):
            item_user_his_mid_mask[idx,idy,:len(y)] = 1.0
            item_user_his_mid[idx,idy,:len(y)] = y
            item_user_his_cat[idx,idy,:len(y)] = seqs_user_cat[idx][idy]

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])
    if use_negsampling:
        return uids, mids, cats, mid_his, cat_his, noclk_mid_his, noclk_cat_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, numpy.array(target), numpy.array(lengths_x)
    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, numpy.array(target), numpy.array(lengths_x)
def eval(sess, test_data, model, model_path, maxlen, user_maxlen, use_negsampling=False):
    import math
    from sklearn import metrics
    y_true = []
    y_pred = []
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    logloss = 0.
    sample_num = 0
    stored_arr = []
    for src, tgt in test_data:
        nums += 1
        if use_negsampling:
            uids, mids, cats, mid_his, cat_his, neg_mid_his, neg_cat_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, target, sl  = prepare_data(src, tgt,maxlen,user_maxlen, use_negsampling)
            prob, loss, acc, aux_loss = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, item_user_his, item_user_his_mask,
                                                             item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, target, sl, neg_mid_his, neg_cat_his])
        else:
            uids, mids, cats, mid_his, cat_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, target, sl = prepare_data(
                src, tgt, maxlen, user_maxlen, use_negsampling)
            prob, loss, acc, aux_loss = model.calculate(sess,
                                                        [uids, mids, cats, mid_his, cat_his, mid_mask, item_user_his,
                                                         item_user_his_mask,
                                                         item_user_his_mid, item_user_his_cat, item_user_his_mid_mask,
                                                         target, sl])
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            sample_num += 1
            # logloss += -1.0*(t*math.log(p)+(1-t)*math.log(1-p))
            y_true.append(t)
            y_pred.append(p)
            stored_arr.append([p, t])
    test_auc = metrics.roc_auc_score(y_true, y_pred)
    # test_f1 = metrics.f1_score(numpy.round(y_true), numpy.round(y_pred))
    Logloss = metrics.log_loss(y_true, y_pred)
    # test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        #model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, Logloss

def train(
        train_file = "../data/Beauty/local_train_sample_sorted_by_time",#change to your path
        test_file = "../data/Beauty/local_test_sample_sorted_by_time",#change to your path
        uid_voc = "../data/Beauty/uid_voc.pkl",#change to your path
        mid_voc = "../data/Beauty/mid_voc.pkl",#change to your path
        cat_voc = "../data/Beauty/cat_voc.pkl",#change to your path
        batch_size = 128,
        maxlen = 20,
        user_maxlen = 50,
        test_iter = 100,
        save_iter = 100,
        model_type = 'DNN',
	    seed = 2,
        use_negsampling=False
):
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)+"_"+str(user_maxlen)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)+"_"+str(user_maxlen)
    if model_type in ["DIEN", "DUMIN"]:
        use_negsampling = True
    else:
        use_negsampling = False
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False, use_negsampling=use_negsampling)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, use_negsampling=use_negsampling)
        n_uid, n_mid, n_cat = train_data.get_n()
        print(n_uid, n_mid, n_cat)
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'SVDPP':
            model = Model_SVDPP(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'GRU4REC':
            model = Model_GRU4REC(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DUMN':
            model = Model_DUMN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, True)
        elif model_type == 'DUMIN':
            model = Model_DUMIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, True)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        # model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        print('test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % eval(sess, test_data, model, best_model_path,maxlen,user_maxlen, use_negsampling))
        sys.stdout.flush()

        start_time = time.time()
        iter = 0
        lr = 0.001
        for itr in range(1):
            print("===============epoch:%d================" % itr)
            loss_sum = 0.0
            accuracy_sum = 0.
            log_loss_sum = 0.
            for src, tgt in train_data:
                if use_negsampling:
                    uids, mids, cats, mid_his, cat_his, neg_mid_his, neg_cat_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, target, sl = prepare_data(src, tgt, maxlen,user_maxlen, use_negsampling)
                    loss, acc, log_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, item_user_his, item_user_his_mask,
                                                         item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, target, sl, lr, neg_mid_his, neg_cat_his])
                else:
                    uids, mids, cats, mid_his, cat_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, target, sl = prepare_data(
                        src, tgt, maxlen, user_maxlen)
                    loss, acc, log_loss = model.train(sess,
                                                      [uids, mids, cats, mid_his, cat_his, mid_mask, item_user_his,
                                                       item_user_his_mask,
                                                       item_user_his_mid, item_user_his_cat, item_user_his_mid_mask,
                                                       target, sl, lr])
                loss_sum += loss
                accuracy_sum += acc
                log_loss_sum += log_loss
                iter += 1
                sys.stdout.flush()
                if (iter % test_iter) == 0:
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- Logloss: %.4f' % \
                        (iter, loss_sum / test_iter, accuracy_sum / test_iter, log_loss_sum / test_iter))
                    print('test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % eval(sess, test_data, model, best_model_path, maxlen,user_maxlen, use_negsampling))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % save_iter) == 0:
                    print('save model iter: %d' %(iter))
                    model.save(sess, model_path+"--"+str(iter))

            lr *= 0.75

def test(
        train_file="../data/Beauty/local_train_sample_sorted_by_time",#change to your path
        test_file="../data/Beauty/local_test_sample_sorted_by_time",#change to your path
        uid_voc="../data/Beauty/uid_voc.pkl",#change to your path
        mid_voc="../data/Beauty/mid_voc.pkl",#change to your path
        cat_voc="../data/Beauty/cat_voc.pkl",#change to your path
        batch_size = 128,
        user_maxlen = 50,
        maxlen = 20,
        model_type = 'DNN',
	    seed = 2,
        use_negsampling=False
):
    model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)+ "_"+str(user_maxlen)
    if model_type in ["DIEN", "DUMIN"]:
        use_negsampling = True
    else:
        use_negsampling = False
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, use_negsampling=use_negsampling)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, use_negsampling=use_negsampling)
        n_uid, n_mid, n_cat = train_data.get_n()
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'SVDPP':
            model = Model_SVDPP(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'GRU4REC':
            model = Model_GRU4REC(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DUMN':
            model = Model_DUMN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, True)
        elif model_type == 'DUMIN':
            model = Model_DUMIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, True)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        model.restore(sess, model_path)
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % eval(sess, test_data, model, model_path, maxlen,user_maxlen, use_negsampling))

if __name__ == '__main__':
    if len(sys.argv) == 4:
        SEED = int(sys.argv[3])
    else:
        SEED = 24
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    approach = 'train'
    model_type = 'DUMIN'
    if approach == 'train':
        train(model_type=model_type, seed=SEED, user_maxlen=20)
    elif approach == 'test':
        test(model_type=model_type, seed=SEED, user_maxlen=25)
    else:
        print('do nothing...')


