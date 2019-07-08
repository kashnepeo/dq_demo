import os
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from ml_rest.ml.regression.preprocessing import *
from ml_rest.ml.regression.predictgenerator import *

trainSet = 0.7
testSet = 0.3


class BaggingClass:
    """
    Name      : BaggingRegressor
    Attribute : None
    Method    : predict, predict_by_cv, save_model
    """

    def __init__(self, filename):
        # 알고리즘 이름
        self._name = 'bagging'

        # 기본 경로
        self._f_path = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)) + "/regression/resource/"
        print("_f_path: ", self._f_path)

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')

        # 원본 데이터 로드
        # data = pd.read_csv(self._f_path + "/regression/resource/regressor_original.csv", sep=",", encoding="utf-8")
        # print(data.head(5))
        # print(data.info())

        # 전처리 클래스 생성
        self.original_filepath = self._f_path
        self.original_filename = filename
        preprocessor = Preprocessing(filepath=self.original_filepath, filename=self.original_filename)

        # 학습 및 레이블(정답) 데이터
        self._x = preprocessor._x
        self._y = preprocessor._y
        self.data = preprocessor.preprocess_df

        # 학습 및 테스트 데이터 분리(7:3)
        n_of_train = int(round(len(self._x) * trainSet))
        n_of_test = int(round(len(self._x) * testSet))
        print("샘플 개수: %d" % len(self._x))
        print("트레이닝셋 갯수: %d, 트레이닝셋: %.2f%%" % (n_of_train, (trainSet * 100)))
        print("테스트셋 갯수: %d, 테스트셋: %.2f%%" % (n_of_test, (testSet * 100)))
        self._x_train = self._x[:n_of_train]
        self._y_train = self._y[:n_of_train]
        self._x_test = self._x[n_of_train:]
        self._y_test = self._x[n_of_train:]
        print("_x_train: ", len(self._x_train))
        print("_y_train: ", len(self._y_train))
        print("_x_test: ", len(self._x_test))
        print("_y_test: ", len(self._y_test))

        # 후처리 클래스 생성
        columns = self.data.columns.tolist()  # 전처리 컬럼 리스트 (후처리시 동일한 컬럼으로 사용)
        maxdate = max(self._x[:, 0])  # 전처리 최대일자 (후처리시 그 다음날 기준으로 date 생성)
        predictgenerator = PredictGeneraotr(columns, maxdate)

        # 모델 선언
        self._model = BaggingRegressor()

        # 모델 학습
        self._model.fit(self._x_train, self._y_train)

        # 그리드 서치 모델
        # self._g_model = None

    # 일반 예측
    def predict(self, save_img=False, show_chart=False):
        # 예측
        y_pred = self._model.predict(self._x_train)
        print("y_pred: ", y_pred, len(y_pred))
        print("self._y_test: ", len(self._y_test))

        # 스코어 정보
        score = r2_score(self._y_train, y_pred)
        print("score: ", score)

        # 리포트 확인
        if hasattr(self._model, 'coef_') and hasattr(self._model, 'intercept_'):
            print(f'Coef = {self._model.coef_}')
            print(f'intercept = {self._model.intercept_}')

        print(f'Score = {score}')

        # 이미지 저장 여부
        if save_img:
            self.save_chart_image(y_pred, show_chart)

        # 예측 값  & 스코어
        return [list(y_pred), score]

    #  CV 예측(Cross Validation)
    def predict_by_cv(self):
        # Regression 알고리즘은 실 프로젝트 상황에 맞게 Cross Validation 구현
        return False

    #  GridSearchCV 예측
    def predict_by_gs(self):
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
        self._g_model = GridSearchCV(BaggingRegressor(), param_grid=param_grid)

        # 그리드 서치 학습
        self._g_model.fit(self._x_train, self._y_train)

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
        # 모델 저장
        if not renew:
            # 처음 저장
            joblib.dump(self._model, self._f_path + f'/model/{self._name}_rg.pkl')
        else:
            # 기존 모델 대체
            if os.path.isfile(self._f_path + f'/model/{self._name}_rg.pkl'):
                os.rename(self._f_path + f'/model/{self._name}_rg.pkl',
                          self._f_path + f'/model/{str(self._name) + str(time.time())}_rg.pkl')
            joblib.dump(self._model, self._f_path + f'/model/{self._name}_rg.pkl')

    # 회귀 차트 저장
    def save_chart_image(self, data, show_chart):
        # 사이즈
        plt.figure(figsize=(15, 10), dpi=100)

        # 레이블
        plt.plot(self._y_test, c='r')

        # 예측 값
        plt.plot(data, c='b')

        # 이미지로 저장
        plt.savefig('./chart_images/tenki-kion-lr.png')

        # 차트 확인(Optional)
        if show_chart:
            plt.show()

    def __del__(self):
        pass
        # del self._x_train, self._x_test, self._y_train, self._y_test, self._x, self._y, self._model


if __name__ == "__main__":
    # 클래스 선언
    classifier = BaggingClass(filename='regressor_original.csv')

    # 분류 실행
    classifier.predict()

    # 분류 실행(이미지 생성& 차트 확인)
    # classifier.predict(save_img=True, show_chart=True)

    # 분류 실행(Cross Validation)
    # classifier.predict_by_cv()

    # 모델 신규
    # classifier.save_model()

    # 모델 갱신
    # classifier.save_model(renew=True)

    # 그리드 서치 실행
    # classifier.predict_by_gs()
