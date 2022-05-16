import sys
import hashlib
import random

def get_cut_time(percent=0.85):
    time_list = []
    fin = open("../data/Beauty/local_all_sample_sorted_by_time", "r")#change to your path
    for line in fin:
        line = line.strip()
        time = float(line.split("\t")[-1])
        time_list.append(time)
    sample_size = len(time_list)
    print(sample_size)
    train_size = int(sample_size * percent)
    time_list = sorted(time_list, key=lambda x: x)
    cut_time = time_list[train_size]
    return cut_time



def split_test_by_time(cut_time):
  fin = open("../data/Beauty/local_all_sample_sorted_by_time", "r")#change to your path
  ftrain = open("../data/Beauty/local_train_sample_sorted_by_time", "w")#change to your path
  ftest = open("../data/Beauty/local_test_sample_sorted_by_time", "w")#change to your path

  for line in fin:
    line = line.strip()
    time = float(line.split("\t")[-1])

    if time <= cut_time:
      print(line, file=ftrain)
    else:
      print(line, file=ftest)


maxlen = 20
user_maxlen = 50
def get_all_samples():
    fin = open("../data/Beauty/jointed-time-new", "r")#change to your path
    # ftrain = open("local_train", "w")
    ftest = open("../data/Beauty/local_all_sample_sorted_by_time", "w") #change to your path

    user_his_items = {}
    user_his_cats = {}
    item_his_users = {}
    line_idx = 0
    for line in fin:
        items = line.strip().split("\t")
        clk = int(items[0])
        user = items[1]
        item_id = items[2]
        dt = items[4]
        cat1 = items[5]
        if user in user_his_items:
            bhvs_items = user_his_items[user][-maxlen:]
        else:
            bhvs_items = []
        if user in user_his_cats:
            bhvs_cats = user_his_cats[user][-maxlen:]
        else:
            bhvs_cats = []

        user_history_clk_num = len(bhvs_items)
        bhvs_items_str = "|".join(bhvs_items)
        bhvs_cats_str  = "|".join(bhvs_cats)

        if item_id in item_his_users:
            item_clk_users = item_his_users[item_id][-user_maxlen:]
        else:
            item_clk_users = []
        item_history_user_num = len(item_clk_users)
        history_users_feats = ";".join(item_clk_users)
        if user_history_clk_num >= 1:    # 8 is the average length of user behavior
            print(items[0] + "\t" + user + "\t" + item_id + "\t" + cat1 +"\t" + bhvs_items_str + "\t" + bhvs_cats_str+ "\t" + history_users_feats+"\t" +dt, file=ftest)
        if clk:
            if user not in user_his_items:
                user_his_items[user] = []
                user_his_cats[user] = []
            user_his_items[user].append(item_id)
            user_his_cats[user].append(cat1)
            if item_id not in item_his_users:
                item_his_users[item_id] = []
            if user_history_clk_num >=1:
                item_bhvs_feat = user+'_'+bhvs_items_str+'_'+bhvs_cats_str
            else:
                item_bhvs_feat = user+'_'+''+'_'+''
            if user_history_clk_num >= 1:
                item_his_users[item_id].append(item_bhvs_feat)
        line_idx += 1


get_all_samples()
cut_time = get_cut_time(percent=0.85)
split_test_by_time(cut_time)

