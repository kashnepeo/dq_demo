import json
import os
import pickle
import re
import warnings
from datetime import date

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocessing():
    def __init__(self, filename, learning, prediction):
        self.cur_dir = os.getcwd()

        self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
        # filename = 'sample2.csv'

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')

        # 원본 데이터 로드
        # data = pd.read_csv(self._f_path + "/classifier/resource/classifier_sample.csv", sep=",", encoding="utf-8")
        today = date.today().strftime('%Y%m%d')

        self.df = pd.read_csv(self._f_path + '/classifier/resource/{}/'.format(today) + filename, sep=",", encoding="ms949")

        # 학습 및 레이블(정답) 데이터 분리
        self._x = self.df[learning]
        self._y = self.df[prediction]

        self.vectorizer = TfidfVectorizer(
            # analyzer='word',
            lowercase=True,
            tokenizer=self.tokenizer_jiana,
            # preprocessor=None,
            # stop_words='english',
            min_df=2,  # 토큰이 나타날 최소 문서 개수로 오타나 자주 나오지 않는 특수한 전문용어 제거에 좋다.
            # ngram_range=(1, 3),
            # vocabulary=set(words.words()),  # nltk의 words를 사용하거나 문서 자체의 사전을 만들거나 선택한다.
            max_features=90000
        )

    def read_stopword(self, cur_dir):
        return pickle.load(open(os.path.join(self._f_path, 'classifier', 'data', 'pklObject', 'stopword.pkl'), 'rb'))

    def write_stopword(self, stopword, stopword_list, cur_dir):
        if stopword not in stopword_list:
            stopword_list.append(stopword)

        # dest = os.path.join(cur_dir, '..\\..\\data', 'pklObject')
        pickle.dump(stopword_list, open(os.path.join(self._f_path, 'classifier', 'data', 'pklObject', 'stopword.pkl'), 'wb'), protocol=4)

    # 공백으로 단어 분리
    def tokenizer(self, text):
        return text.split()

    # 형태소 분석기를 이용한 단어 분리
    def tokenizer_okt(self, text):
        if not text:
            text = '.'
        return [token for (token, tag) in self.okt.pos(text, norm=True, stem=True) if (tag == 'Noun' or tag == 'Adjective' or tag == 'Adverb') and token not in self.read_stopword(self.cur_dir) and len(token) > 1]

        # 형태소 분석기를 이용한 단어 분리

    def tokenizer_jiana(self, text):
        if not text:
            text = '.'
        url = 'http://localhost:9090/analysis'
        headers = {'Content-Type': 'application/json;charset=UTF-8', 'accept-charset': 'UTF-8'}
        # tag_info = 'ncn,nq_loc,ncp,pv,pa'
        # tag_info = 'nc,ncn,ncp,nq,nq_per,nq_loc,nq_juso,nq_etc,pv,pa,ma'
        tag_info = 'nc,ncn,ncp,nq'
        response = requests.post(url=url, data=json.dumps({'rawData': text, 'tagInfo': tag_info, 'wsNum': 3}), headers=headers, timeout=5)
        response_str = json.loads(response.text)

        return [re.sub('[0-9]{1,4}([_])', ' ', element).replace('|', '').strip() for element in response_str['result'].strip("[]").split(", ")]

    def keyword_vectorizer(self, x_train, x_test):
        X_train_tfidf_vector = self.vectorizer.fit_transform(x_train)
        vocab = self.vectorizer.get_feature_names()
        print('feature counts : ', len(vocab))

        X_test_tfidf_vector = self.vectorizer.transform(x_test)

        dist = np.sum(X_train_tfidf_vector, axis=0)
        # print(self.vectorizer.vocabulary_)
        # {'행복하': 490, '여행': 263, '시작': 217, '인터넷 면세점': 342, '시이': 216, '드리': 93, '지금': 407, '취소': 434, '인도': 335, '그렇': 47, '우선': 300, '고객': 24,  ...}

        vec2dict_list = self.vec2dict(X_test_tfidf_vector, vocab)
        return (X_train_tfidf_vector, X_test_tfidf_vector, vocab, dist, vec2dict_list)

    def vec2dict(self, vector, vocab):
        result_list = []
        for vec_array in vector.toarray().tolist():
            vec_dict = {}
            for idx, row in enumerate(vec_array):
                if row != 0:
                    vec_dict[vocab[idx]] = row
            result_list.append(vec_dict)

        """
        print(result_list)
        [{'감사하': 0.23119708875691353, '공치': 0.2803828859329438, '기다리': 0.2831901925747923, .... '해주시': 0.12469584761697167, '행복하': 0.038231980697211065, '확인': 0.11408861631246986}]
        """
        return result_list
