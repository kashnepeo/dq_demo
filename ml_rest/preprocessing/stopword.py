import pickle
import os


def read_stopword(cur_dir):
    return pickle.load(open(os.path.join(cur_dir, 'data', 'pklObject', 'stopword.pkl'), 'rb'))


def write_stopword(stopword, stopword_list, cur_dir):
    if stopword not in stopword_list:
        stopword_list.append(stopword)

    dest = os.path.join(cur_dir, 'data', 'pklObject')
    pickle.dump(stopword_list, open(os.path.join(dest, 'stopword.pkl'), 'wb'), protocol=4)


if __name__ == '__main__':

    cur_dir = os.getcwd()
    stopword = '네네'

    try:
        write_stopword(stopword, read_stopword(cur_dir), cur_dir)
    except FileNotFoundError:
        print('First write stopword')
        write_stopword(stopword, [], cur_dir)
