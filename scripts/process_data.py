import sys
import random
import time

def process_meta(file):
    fi = open(file, "r")
    fo = open("../data/Beauty/item-info", "w") #change to your path
    for line in fi:
        obj = eval(line)
        cat = obj["categories"][0][-1]
        print(obj["asin"] + "\t" + cat, file=fo)

def process_reviews(file):
    fi = open(file, "r")
    user_map = {}
    fo = open("../data/Beauty/reviews-info", "w") #change to your path
    for line in fi:
        obj = eval(line)
        userID = obj["reviewerID"]
        itemID = obj["asin"]
        rating = obj["overall"]
        time = obj["unixReviewTime"]
        print(userID + "\t" + itemID + "\t" + str(rating) + "\t" + str(time), file=fo)


def data_stats():
    import numpy
    f_rev = open("../data/Beauty/reviews-info", "r") #change to your path
    user_map = {}
    item_map = {}
    sample_list = []
    num = 0
    for line in f_rev:
        num += 1
        line = line.strip()
        items = line.split("\t")
        sample_list.append((line, float(items[-1])))
        if items[0] not in user_map:
            user_map[items[0]] = []
        user_map[items[0]].append(items[1])
        if items[1] not in item_map:
            item_map[items[1]] = []
        item_map[items[1]].append(items[0])
    print("review num:{}".format(num))
    print("user num:{}".format(len(user_map)))
    print("item num:{}".format(len(item_map)))
    user_length = [len(user_map[k]) for k in user_map]
    item_length = [len(item_map[k]) for k in item_map]
    user_length_max = numpy.max(user_length)
    item_length_max = numpy.max(item_length)
    user_length_avg = numpy.mean(user_length)
    item_length_avg = numpy.mean(item_length)
    print(user_length_max,item_length_max,user_length_avg,item_length_avg)

def manual_join_as_time():
    f_rev = open("../data/Beauty/reviews-info", "r")#change to your path
    # user_map = {}
    item_list = []
    sample_list = []
    for line in f_rev:
        line = line.strip()
        items = line.split("\t")
        sample_list.append((line,float(items[-1]))) #[((u,i,r,t), t)....]
        #loctime = time.localtime(float(items[-1]))
        #items[-1] = time.strftime('%Y-%m-%d', loctime)
        # if items[0] not in user_map:
        #     user_map[items[0]]= []
        # user_map[items[0]].append(("\t".join(items), float(items[-1])))
        item_list.append(items[1])
    print("sample size:{}".format(len(item_list)))
    sample_list = sorted(sample_list, key=lambda x:x[1])
    f_meta = open("../data/Beauty/item-info", "r") #change to your path
    meta_map = {}
    for line in f_meta:
        arr = line.strip().split("\t")
        if arr[0] not in meta_map:
            meta_map[arr[0]] = arr[1]
            arr = line.strip().split("\t")
    fo = open("../data/Beauty/jointed-time-new", "w") #change to your path
    for line in sample_list:
        items = line[0].split("\t")
        asin = items[1]
        j = 0
        while True: #neg sample
            asin_neg_index = random.randint(0, len(item_list) - 1)
            asin_neg = item_list[asin_neg_index]
            if asin_neg == asin:
                continue
            items[1] = asin_neg
            print("0" + "\t" + "\t".join(items) + "\t" + meta_map[asin_neg], file=fo)
            j += 1
            if j == 1:             #negative sampling frequency
                break
        if asin in meta_map:
            print("1" + "\t" + line[0] + "\t" + meta_map[asin], file=fo)
        else:
            print("1" + "\t" + line[0] + "\t" + "default_cat", file=fo)



process_meta('../Datasets/Beauty/meta_Beauty.json')#change to your path
process_reviews('../Datasets/Beauty/Beauty_5.json')#change to your path
manual_join_as_time()
data_stats()
