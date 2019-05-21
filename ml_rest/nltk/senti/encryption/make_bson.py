import bson
import json
import pickle


def make_bson():
    with open('../data/senti_word.json', encoding='utf-8', mode='r') as f:
        origin_data = json.load(f)

    return_list = list()
    for data in origin_data:
        return_list.append(bson.dumps(data))

    with open('../data/senti_word.bson', mode='wb') as f2:
        pickle.dump(return_list, f2, pickle.HIGHEST_PROTOCOL)


def read_bson():
    with open('../data/senti_word.bson', mode='rb') as f3:
        pickle_list = pickle.load(f3)

    for data in pickle_list:
        print(bson.loads(data))
