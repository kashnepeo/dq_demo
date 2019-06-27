import os
import pickle
import warnings

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocessing():
    def __init__(self, filename):
        self.okt = Okt()
        self.cur_dir = os.getcwd()

        self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
        # filename = 'sample2.csv'

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')

        # 원본 데이터 로드
        # data = pd.read_csv(self._f_path + "/classifier/resource/classifier_sample.csv", sep=",", encoding="utf-8")
        self.df = pd.read_csv(self._f_path + '/classifier/resource/' + filename, sep=",", encoding="ms949")

        # 학습 및 레이블(정답) 데이터 분리
        self._x = self.df["STT_CONT"]
        self._y = self.df["CALL_L_CLASS_CD"]

        self.vectorizer = TfidfVectorizer(
            # analyzer='word',
            lowercase=True,
            tokenizer=self.tokenizer,
            # preprocessor=None,
            # stop_words='english',
            min_df=2,  # 토큰이 나타날 최소 문서 개수로 오타나 자주 나오지 않는 특수한 전문용어 제거에 좋다.
            ngram_range=(1, 3),
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

    def keyword_vectorizer(self, x_train, x_test):

        X_train_tfidf_vector = self.vectorizer.fit_transform(x_train)
        vocab = self.vectorizer.get_feature_names()
        print('feature counts : ', len(vocab))

        # X_test_tfidf_vector = pipeline.fit_transform(X_test)
        X_test_tfidf_vector = self.vectorizer.transform(x_test)

        dist = np.sum(X_train_tfidf_vector, axis=0)
        for (tag, cnt) in zip(vocab, dist):
            print(tag, cnt)

        print((X_train_tfidf_vector, X_test_tfidf_vector, vocab, dist))
        return (X_train_tfidf_vector, X_test_tfidf_vector, vocab, dist)
