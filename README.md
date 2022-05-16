# Deep User Multi-Interest Network
our code is implemented based on [DUMN](https://github.com/hzzai/DUMN)
## Running
We use Tensorflow1.14 and python3.6
## Prepare Data
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz

gzip -d reviews_Beauty_5.json.gz

wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz

gzip -d meta_Beauty.json.gz

python ./script/process_data.py meta_Beauty.json reviews_Beauty_5.json

python ./script/local_aggretor_time.py 

python ./script/generate_voc.py 

## Train Model
python ./script/train.py
