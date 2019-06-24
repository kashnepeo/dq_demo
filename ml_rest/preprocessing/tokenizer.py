from konlpy.tag import Okt
from stopword import read_stopword
import os

okt = Okt()
cur_dir = os.getcwd()
print(okt)


# 공백으로 단어 분리
def tokenizer(text):
    return text.split()


def tokenizer_okt(text):
    if not text:
        text = '.'
    return [token for (token, tag) in okt.pos(text, norm=True, stem=True) if (tag == 'Noun' or tag == 'Adjective') and token not in read_stopword(cur_dir)]
