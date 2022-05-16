import numpy
import json
import _pickle as pkl
import random

import gzip

import shuffle

def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())


def load_dict(filename):
    try:
        data = pkl.load(open(filename, "rb"))
        return data
    except Exception as e:
        print(e)
        exit(0)


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class DataIterator:

    def __init__(self, source,
                 uid_voc,
                 mid_voc,
                 cat_voc,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None,
                 use_negsampling=False):
        self.use_negsampling = use_negsampling
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
        self.source_dicts = []
        for source_dict in [uid_voc, mid_voc, cat_voc]:
            self.source_dicts.append(load_dict(source_dict))

        f_meta = open("../data/Beauty/item-info", "r")#change to your path
        meta_map = {}#item名:cate名 的映射
        for line in f_meta:
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1]
        self.meta_id_map ={} # item_id:cate_id的映射
        for key in meta_map:
            val = meta_map[key]
            if key in self.source_dicts[1]:
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx

        f_review = open("../data/Beauty/reviews-info", "r")#change to your path
        self.mid_list_for_random = [] #所有的review中item名对应的item_id
        for line in f_review:
            arr = line.strip().split("\t")
            tmp_idx = 0
            if arr[1] in self.source_dicts[1]:
                tmp_idx = self.source_dicts[1][arr[1]]
            self.mid_list_for_random.append(tmp_idx)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source= shuffle.main(self.source_orig, temporary=True)
        else:
            self.source.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
                mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
                cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0
                tmp = []
                for fea in ss[4].split("|"):
                    m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                    tmp.append(m)
                mid_list = tmp

                tmp1 = []
                for fea in ss[5].split("|"):
                    c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                    tmp1.append(c)
                cat_list = tmp1
                item_bhvs_uid_feats = []
                item_bhvs_id_feats = []
                item_bhvs_cat_feats = []
                for bhvs in ss[6].split(";"):
                    if bhvs.strip() =="":
                        break
                    arr = bhvs.split("_")
                    if arr[1].strip() == "":
                        continue
                    bhv_uid = self.source_dicts[0][arr[0]] if arr[0] in self.source_dicts[0] else 0
                    item_bhvs_uid_feats.append(bhv_uid)
                    if arr[1].strip() != "":
                        id_tmp_list = []
                        for fea in arr[1].split("|"):
                            m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                            id_tmp_list.append(m)
                        item_bhvs_id_feats.append(id_tmp_list)
                    if arr[2].strip() != "":
                        cat_tmp_list = []
                        for fea in arr[2].split("|"):
                            c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                            cat_tmp_list.append(c)
                        item_bhvs_cat_feats.append(cat_tmp_list)

                # read from source file and map to word index

                #if len(mid_list) > self.maxlen:
                #    continue
                if self.minlen != None:
                    if len(mid_list) <= self.minlen:
                        continue
                if self.skip_empty and (not mid_list):
                    continue
                if self.use_negsampling:
                    # random neg sample
                    noclk_mid_list = []
                    noclk_cat_list = []
                    for pos_mid in mid_list:
                        noclk_tmp_mid = []
                        noclk_tmp_cat = []
                        noclk_index = 0
                        while True:
                            noclk_mid_indx = random.randint(0, len(self.mid_list_for_random) - 1)
                            noclk_mid = self.mid_list_for_random[noclk_mid_indx]
                            if noclk_mid == pos_mid:
                                continue
                            noclk_tmp_mid.append(noclk_mid)
                            noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
                            noclk_index += 1
                            if noclk_index >= 3:
                                break
                        noclk_mid_list.append(noclk_tmp_mid)
                        noclk_cat_list.append(noclk_tmp_cat)
                if self.use_negsampling:
                    source.append([uid, mid, cat, mid_list, cat_list, item_bhvs_uid_feats, item_bhvs_id_feats,item_bhvs_cat_feats, noclk_mid_list, noclk_cat_list])
                else:
                    source.append([uid, mid, cat, mid_list, cat_list, item_bhvs_uid_feats, item_bhvs_id_feats,item_bhvs_cat_feats])
                target.append([float(ss[0]), 1-float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target


if __name__ == '__main__':
    train_file = "../data/Beauty/local_train_sample_sorted_by_time" #test dataloader
    test_file = "../data/Beauty/local_test_sample_sorted_by_time"
    uid_voc = "../data/Beauty/uid_voc.pkl"
    mid_voc = "../data/Beauty/mid_voc.pkl"
    cat_voc = "../data/Beauty/cat_voc.pkl"
    train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, 2, 100, shuffle_each_epoch=False)
    num = 0
    for src, tgt in train_data:
        print(src)
        print(tgt)
