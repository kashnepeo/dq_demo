import os
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import ARDRegression
from sklearn.metrics import r2_score


class ARDClass:
    """
    Name      : ARDRegression
    Attribute : None
    Method    : predict, predict_by_cv, save_model
    """

    def __init__(self):
        # 알고리즘 이름
        self._name = 'ard'

        # 기본 경로
        self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')
        print(123)
        # 원본 데이터 로드
        data = pd.read_csv(self._f_path + "/regression/resource/regression_sample.csv", sep=",", encoding="utf-8")

        # 학습 및 테스트 데이터 분리
        self._x = (data["year"] <= 2017)
        self._y = (data["year"] >= 2018)

        # 학습 데이터 분리
        self._x_train, self._y_train = self.preprocessing(data[self._x])
        # 테스트 데이터 분리
        self._x_test, self._y_test = self.preprocessing(data[self._y])

        # 모델 선언
        self._model = ARDRegression(normalize=True)

        # 모델 학습
        self._model.fit(self._x_train, self._y_train)

    # 데이터 전처리
    def preprocessing(self, data):
        # 학습
        x = []
        # 레이블
        y = []
        # 기준점(7일)
        base_interval = 7
        # 기온
        temps = list(data["temperature"])

        for i in range(len(temps)):
            if i < base_interval:
                continue
            y.append(temps[i])

            xa = []

            for p in range(base_interval):
                d = i + p - base_interval
                xa.append(temps[d])
            x.append(xa)
        return x, y

    # 일반 예측
    def predict(self, save_img=False, show_chart=False):
        # 예측
        y_pred = self._model.predict(self._x_test)

        # 스코어 정보
        score = r2_score(self._y_test, y_pred)

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
        pass

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
        del self._x_train, self._x_test, self._y_train, self._y_test, self._x, self._y, self._model


if __name__ == "__main__":
    # 클래스 선언
    classifier = ARDClass()

    # 분류 실행
    # classifier.predict()

    # 분류 실행(이미지 생성& 차트 확인)
    classifier.predict(save_img=True, show_chart=True)

    # 분류 실행(Cross Validation)
    # classifier.predict_by_cv()

    # 모델 신규
    # classifier.save_model()

    # 모델 갱신
    # classifier.save_model(renew=True)
