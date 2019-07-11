import os
import time
import warnings

import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB

from .preprocessing import Preprocessing


# from ml_rest.ml.classifier.preprocessing import Preprocessing


class BernouliNBClass:
    """
    Name      : BernouliNB
    Attribute : None
    Method    : predict, predict_by_cv, save_model
    """

    def __init__(self, params, filename):
        # 알고리즘 이름
        self._name = 'bernoulinb'
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
        self.X_train_tfidf_vector, self.X_test_tfidf_vector, self.vocab, self.dist, self.result_a = preprocessor.keyword_vectorizer(x_train=self._x_train, x_test=self._x_test)

        # 모델 선언
        self._model = BernoulliNB()

        # 전처리 데이터를 이용한 모델 학습
        test = self._model.fit(self.X_train_tfidf_vector, self._y_train)
        # print(test.decision_function(self._x_test))

    # 일반 예측
    def predict(self, config):
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

        print('call_end_time', len(call_end_time))
        print('self.result_a', len(self.result_a))

        feature_data = self.make_feature_data(config)
        self.output = pd.DataFrame(
            data={'recordkey': recordkey, 'stt_cont': '', 'call_l_class_cd': call_l_class_cd, 'call_m_class_cd': call_m_class_cd, 'call_start_time': call_start_time, 'call_end_time': call_end_time, 'predict': self.y_pred,
                  'keywords': self.result_a})

        # 분류 결과 file write
        self.output.to_csv(self._f_path + f'/classifier/csv/result_{self._name}.csv', index=False, quoting=3, escapechar='\\')
        print('write')
        # 분석 feature file write
        pd.DataFrame(self.dist, columns=self.vocab).to_csv(self._f_path + f'/classifier/csv/features_{self._name}.csv', index=False, quoting=3)

        # 스코어 리턴, 레포트 정보, 테스트셋 분석결과
        return score, report_df, self.output, feature_data

    #  CV 예측(Cross Validation)
    def predict_by_cv(self):
        cv = KFold(n_splits=5, shuffle=True)
        # CV 지원 여부
        if hasattr(self._model, "score"):
            cv_score = cross_val_score(self._model, self._x, self._y, cv=cv)
            # 스코어 확인
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

    def make_feature_data(self, config):
        cat_dict = {}
        for vect, pred in zip(self.result_a, self.y_pred):
            print(pred, vect)
            if pred in cat_dict.keys():
                cat_dict[pred].update(vect)
            else:
                cat_dict[pred] = vect
        print(cat_dict)
        for key, value in cat_dict.items():
            cat_dict[key] = sorted(value.items(), key=(lambda x: x[1]), reverse=True)[:20]

        print(cat_dict)
        feature_data = []
        for key, value in cat_dict.items():
            for each_word in value:
                feature_data.append(
                    {"x": str(each_word[0]), "value": float(each_word[1]), "category": config[str(key)]}
                )
        print(feature_data)
        return feature_data


if __name__ == "__main__":
    # 클래스 선언
    classifier = BernouliNBClass({'subject': 'a', 'classifier_algorithm': 'c', 'model_save': 'aaa', 'learning_coloumn': 'STT_CONT', 'prediction_coloumn': 'CALL_L_CLASS_CD'}, 'refined_call_112.csv')

    # 분류 실행
    classifier.predict()

    # 분류 실행(Cross Validation)
    # classifier.predict_by_cv()

    # 모델 신규
    classifier.save_model()

    # 모델 갱신
    # classifier.save_model(renew=True)

    # 그리드 서치 실행
    # classifier.predict_by_gs()
