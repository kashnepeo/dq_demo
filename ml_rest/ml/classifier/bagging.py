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


class BaggingClass:
    """
    Name      : BaggingClassfier
    Attribute : None
    Method    : predict, predict_by_cv, save_model
    """

    def __init__(self):
        # 알고리즘 이름
        self._name = 'bagging'
        # 기본 경로
        self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')

        # 원본 데이터 로드
        data = pd.read_csv(self._f_path + "/classifier/resource/classifier_sample.csv", sep=",", encoding="utf-8")

        # 학습 및 레이블(정답) 데이터 분리
        self._x = data.drop("quality", axis=1)
        self._y = data["quality"]

        # 학습 데이터 및 테스트 데이터 분리
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(self._x, self._y, test_size=0.2,
                                                                                    shuffle=True,
                                                                                    random_state=42)
        # 모델 선언
        self._model = BaggingClassifier()

        # 모델 학습
        self._model.fit(self._x_train, self._y_train)

        # 그리드 서치 모델
        self._g_model = None

    # 일반 예측
    def predict(self):
        # 예측
        y_pred = self._model.predict(self._x_test)

        # 리포트 출력
        print(classification_report(self._y_test, y_pred))

        score = accuracy_score(self._y_test, y_pred)

        # 스코어 확인
        print(f'Score = {score}')
        # 스코어 리턴
        return score

    #  CV 예측(Cross Validation)
    def predict_by_cv(self):
        # KFold 선언
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
        self._g_model.fit(self._x_train, self._y_train)
        
        # 파라미터 모두 출력
        print(self._g_model.param_grid)
        # 베스트 스코어
        print(self._g_model.best_score_)
        # 베스트 파라미터
        print(self._g_model.best_params_)
        # 전체 결과 출력
        print(self._g_model.cv_results_)

        return dict(gs_all_params=self._g_model.param_grid, gs_best_score=self._g_model.best_score_, gs_best_param=self._g_model.best_params_)

    # 모델 저장 및 갱신
    def save_model(self, renew=False):
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
        del self._x_train, self._x_test, self._y_train, self._y_test, self._x, self._y, self._model


if __name__ == "__main__":
    # 클래스 선언
    classifier = BaggingClass()

    # 분류 실행
    classifier.predict()

    # 분류 실행(Cross Validation)
    # classifier.predict_by_cv()

    # 모델 신규
    classifier.save_model()

    # 모델 갱신
    # classifier.save_model(renew=True)

    # classifier.predict_by_gs()
