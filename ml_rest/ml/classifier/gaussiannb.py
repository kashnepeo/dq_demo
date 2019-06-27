import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import *

# from ml_rest.ml.classifier.preprocessing import Preprocessing
from .preprocessing import Preprocessing


class GaussianNBClass:
    """
    Name      : GaussianNB
    Attribute : None
    Method    : predict, predict_by_cv, save_model
    """

    def __init__(self, params, filename):
        # 알고리즘 이름
        self._name = 'gaussiannb'
        # 기본 경로
        self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')

        subject = params['subject']
        classifier_algorithm = params['classifier_algorithm']
        model_save = params['model_save']
        learning_coloumn = params['learning_coloumn']
        prediction_coloumn = params['prediction_coloumn']

        print(model_save, subject)

        # 전처리 클래스 생성
        preprocessor = Preprocessing(filename=filename)

        # 학습 및 레이블(정답) 데이터 분리
        self._x = preprocessor._x
        self._y = preprocessor._y
        self.data = preprocessor.df

        # 원본 데이터 로드
        # data = pd.read_csv(os.path.abspath(os.path.join(self._f_path, '../', filename)), sep=",", encoding="ms949")
        # print('data : ', data.head(10))

        # 학습 데이터 및 테스트 데이터 분리
        # self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(self._x, self._y, test_size=0.2, shuffle=True, random_state=42)
        # self._x_train = self.data.loc[:78, 'STT_CONT'].values
        self._x_train = self.data.loc[:78, learning_coloumn].values
        # self._y_train = self.data.loc[:78, 'CALL_L_CLASS_CD'].values
        self._y_train = self.data.loc[:78, prediction_coloumn].values
        # self._x_test = self.data.loc[34:, 'STT_CONT'].values
        self._x_test = self.data.loc[34:, learning_coloumn].values
        # self._y_test = self.data.loc[34:, 'CALL_L_CLASS_CD'].values
        self._y_test = self.data.loc[34:, prediction_coloumn].values

        # 전처리 데이터 로드
        self.X_train_tfidf_vector, self.X_test_tfidf_vector, self.vocab, self.dist = preprocessor.keyword_vectorizer(x_train=self._x_train, x_test=self._x_test)

        # 모델 선언
        self._model = SVC()

        # 전처리 데이터를 이용한 모델 학습
        self._model.fit(self.X_train_tfidf_vector, self._y_train)

    # 일반 예측
    def predict(self):
        # 예측
        self.y_pred = self._model.predict(self.X_test_tfidf_vector)

        # 리포트 출력
        report_str = classification_report(self._y_test, self.y_pred)
        print(report_str)

        score = accuracy_score(self._y_test, self.y_pred)

        # 스코어 확인
        print(f'Score = {score}')

        # 개행 기준으로 레포트 문자열을 자름
        split_list = [x.split() for x in report_str.split('\n') if x]

        # 레포트 헤더 생성
        header = [['class'] + value for idx, value in enumerate(split_list) if idx == 0][0]

        # class precision recall f1-score support 의 value list
        value_list = [x for x in split_list if 'precision' not in x and 'accuracy' not in x and 'avg' not in x]

        # 레포트 파싱 완료 (DataFrame)
        report_df = pd.DataFrame(value_list, columns=header)
        # print(report_df['class'].values)
        # print(report_df['precision'].values)
        # print(report_df['recall'].values)
        # print(report_df['f1-score'].values)
        # print(report_df['support'].values)

        recordkey = self.data.loc[34:, "RECORDKEY"]
        call_l_class_cd = self.data.loc[34:, "CALL_L_CLASS_CD"]
        call_m_class_cd = self.data.loc[34:, "CALL_M_CLASS_CD"]
        call_start_time = self.data.loc[34:, "CALL_START_TIME"]
        call_end_time = self.data.loc[34:, "CALL_END_TIME"]

        self.output = pd.DataFrame(
            data={'recordkey': recordkey, 'stt_cont': '', 'call_l_class_cd': call_l_class_cd, 'call_m_class_cd': call_m_class_cd, 'call_start_time': call_start_time, 'call_end_time': call_end_time, 'predict': self.y_pred})

        # 스코어 리턴, 레포트 정보, 테스트셋 분석결과
        return score, report_df, self.output

    #  CV 예측(Cross Validation)
    def predict_by_cv(self):
        cv = KFold(n_splits=5, shuffle=True)
        # CV 지원 여부
        if hasattr(self._model, "score"):
            cv_score = np.mean(cross_val_score(self._model, self.X_train_tfidf_vector, self._y_train, cv=cv))
            print(f'Score = {cv_score}')
            # 스코어 리턴
            return cv_score
        else:
            raise Exception('Not Support CrossValidation')

    #  GridSearchCV 예측
    def predict_by_gs(self):
        pass

    # 모델 저장 및 갱신
    def save_model(self, renew=False):

        # 분류 결과 file write
        self.output.to_csv(self._f_path + f'/classifier/csv/result_{self._name}.csv', index=False, quoting=3, escapechar='\\')

        # 분석 feature file write
        pd.DataFrame(self.dist, columns=self.vocab).to_csv(self._f_path + f'/classifier/csv/features_{self._name}.csv', index=False, quoting=3)

        # 모델 저장
        if not renew:
            # 처음 저장
            joblib.dump(self._model, self._f_path + f'/model/{self._name}.pkl')
        else:
            # 기존 모델 대체
            if os.path.isfile(self._f_path + f'/model/{self._name}.pkl'):
                os.rename(self._f_path + f'/model/{self._name}.pkl', self._f_path + f'/model/{str(self._name) + str(time.time())}.pkl')
                joblib.dump(self._model, self._f_path + f'/model/{self._name}.pkl')

    def __del__(self):
        del self._x_train, self._x_test, self._y_train, self._y_test, self._x, self._y, self._model


if __name__ == "__main__":
    # 클래스 선언
    classifier = GaussianNBClass(filename='refined_call_112.csv')

    # 분류 실행
    classifier.predict()

    # 분류 실행(Cross Validation)
    classifier.predict_by_cv()

    # 모델 신규
    # classifier.save_model()

    # 모델 갱신
    classifier.save_model(renew=True)

    # 그리드 서치 실행
    # classifier.predict_by_gs()
