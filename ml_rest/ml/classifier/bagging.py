import os
import time
import warnings

import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from ml_rest.ml.classifier.preprocessing import Preprocessing


class BaggingClass:
    """
    Name      : BaggingClassfier
    Attribute : None
    Method    : predict, predict_by_cv, save_model
    """

    def __init__(self, filename):
        # 알고리즘 이름
        self._name = 'bagging'
        # 기본 경로
        self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')

        # 전처리 클래스 생성
        preprocessor = Preprocessing(filename=filename)

        self._x = preprocessor._x
        self._y = preprocessor._y
        self.data = preprocessor.df

        # 원본 데이터 로드
        # data = pd.read_csv(self._f_path + "/classifier/resource/classifier_sample.csv", sep=",", encoding="utf-8")

        # 학습 및 레이블(정답) 데이터 분리
        # self._x = data.drop("quality", axis=1)
        # self._y = data["quality"]

        # 학습 데이터 및 테스트 데이터 분리
        # self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(self._x, self._y, test_size=0.2,
        #                                                                             shuffle=True,
        #                                                                             random_state=42)

        # 학습 데이터 및 테스트 데이터 분리
        self._x_train = self.data.loc[:78, 'STT_CONT'].values
        self._y_train = self.data.loc[:78, 'CALL_L_CLASS_CD'].values
        self._x_test = self.data.loc[34:, 'STT_CONT'].values
        self._y_test = self.data.loc[34:, 'CALL_L_CLASS_CD'].values
        print("_x_train: ", self._x_train)
        print("_y_train: ", self._y_train)
        print("_x_test: ", self._x_test)
        print("_y_test: ", self._y_test)

        self.X_train_tfidf_vector, self.X_test_tfidf_vector, self.vocab, self.dist = preprocessor.keyword_vectorizer(
            x_train=self._x_train, x_test=self._x_test)
        print("X_train_tfidf_vector: ", self.X_train_tfidf_vector)
        print("X_test_tfidf_vector: ", self.X_test_tfidf_vector)
        print("vocab: ", self.vocab)
        print("dist: ", self.dist)

        # 모델 선언
        self._model = BaggingClassifier()

        # 모델 학습
        self._model.fit(self.X_train_tfidf_vector, self._y_train)

        # 그리드 서치 모델
        self._g_model = None

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

        recordKey = self.data.loc[34:, "RECORDKEY"]
        call_l_class_cd = self.data.loc[34:, "CALL_L_CLASS_CD"]
        call_m_class_cd = self.data.loc[34:, "CALL_M_CLASS_CD"]
        call_start_time = self.data.loc[34:, "CALL_START_TIME"]
        call_end_time = self.data.loc[34:, "CALL_END_TIME"]

        self.output = pd.DataFrame(data={'recordkey': recordKey, 'stt_cont': '', 'call_l_class_cd': call_l_class_cd,
                                         'call_m_class_cd': call_m_class_cd, 'call_start_time': call_start_time,
                                         'call_end_time': call_end_time, 'predict': self.y_pred})

        # 스코어 리턴
        return score, report_df, self.output

    #  CV 예측(Cross Validation)
    def predict_by_cv(self):
        # KFold 선언
        cv = KFold(n_splits=5, shuffle=True)

        # CV 지원 여부
        if hasattr(self._model, "score"):
            cv_score = cross_val_score(self._model, self.X_train_tfidf_vector, self._y_train, cv=cv)
            # 스코어 확인
            print(f'Score = {cv_score}')
            # 스코어 리턴
            return cv_score
        else:
            raise Exception('Not Support CrossValidation')

    #  GridSearchCV 예측
    def predict_by_gs(self):
        # KFold 선언
        cv = KFold(n_splits=5, shuffle=True)

        # 그리드 서치 Params
        param_grid = {
            # 모형 개수
            'n_estimators': [5, 10, 15],
            # 데이터 중복 여부
            'bootstrap': [True, False],
            # 차원 중복 여부
            'bootstrap_features': [True, False],
            # 독립 변수 차원 비율
            'max_samples': [0.6, 0.8, 1.0]
        }

        # 그리드 서치 초기화
        self._g_model = GridSearchCV(BaggingClassifier(), param_grid=param_grid, cv=cv)

        # 그리드 서치 학습
        self._g_model.fit(self.X_train_tfidf_vector, self._y_train)

        # 파라미터 모두 출력
        print(self._g_model.param_grid)
        # 베스트 스코어
        print(self._g_model.best_score_)
        # 베스트 파라미터
        print(self._g_model.best_params_)
        # 전체 결과 출력
        print(self._g_model.cv_results_)

        return dict(gs_all_params=self._g_model.param_grid, gs_best_score=self._g_model.best_score_,
                    gs_best_param=self._g_model.best_params_)

    # 모델 저장 및 갱신
    def save_model(self, renew=False):

        # 분류 결과 file write
        self.output.to_csv(self._f_path + f'/classifier/csv/result_{self._name}.csv', index=False, quoting=3,
                           escapechar='\\')

        # 분석 feature file write
        pd.DataFrame(self.dist, columns=self.vocab).to_csv(self._f_path + f'/classifier/csv/features_{self._name}.csv',
                                                           index=False, quoting=3)

        # 모델 저장
        if not renew:
            # 처음 저장
            joblib.dump(self._model, self._f_path + f'/model/{self._name}.pkl')
        else:
            # 기존 모델 대체
            if os.path.isfile(self._f_path + f'/model/{self._name}.pkl'):
                os.rename(self._f_path + f'/model/{self._name}.pkl',
                          self._f_path + f'/model/{str(self._name) + str(time.time())}.pkl')
            joblib.dump(self._model, self._f_path + f'/model/{self._name}.pkl')

    def __del__(self):
        del self._x_train, self._x_test, self._y_train, self._y_test, self._x, self._y, self._model, self.X_train_tfidf_vector, self.X_test_tfidf_vector, self.vocab, self.dist, self.output


if __name__ == "__main__":
    # 클래스 선언
    classifier = BaggingClass(filename='sample2.csv')

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
