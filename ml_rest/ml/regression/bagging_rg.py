import time

import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# from ml_rest.ml.regression.preprocessing import *
# from ml_rest.ml.regression.predictgenerator import *
from .predictgenerator import *

trainSet = 0.7
testSet = 0.3


class BaggingClass:
    """
    Name      : BaggingRegressor
    Attribute : None
    Method    : predict, predict_by_cv, save_model
    """

    def __init__(self, params, filename):
        print("================ BaggingRegressor __init__ Start =============")
        # 알고리즘 이름
        self._name = 'bagging'
        # 기본 경로
        self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
        self.params = params
        self.filename = filename
        print("params: ", self.params)
        print("filename: ", self.filename)

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')

        # 원본 데이터 로드
        self.data = pd.read_csv(self._f_path + "/regression/csv/" + filename, sep=",", encoding="utf-8")
        print(self.data.head(5))
        print(self.data.info())

        # 학습 및 레이블(정답) 데이터
        self.learning_column = params['learning_column']
        self.prediction_column = params['prediction_column']
        self._x = self.data[self.learning_column].values
        self._y = self.data[self.prediction_column].values

        # 학습 및 테스트 데이터 분리(7:3)
        n_of_train = int(round(len(self._x) * trainSet))
        n_of_test = int(round(len(self._x) * testSet))
        print("샘플 개수: %d" % len(self._x))
        print("트레이닝셋 갯수: %d, 트레이닝셋: %.2f%%" % (n_of_train, (trainSet * 100)))
        print("테스트셋 갯수: %d, 테스트셋: %.2f%%" % (n_of_test, (testSet * 100)))
        self._x_train = self._x[:n_of_train].reshape(-1, 1)
        self._y_train = self._y[:n_of_train].reshape(-1, 1)
        self._x_test = self._x[n_of_test:].reshape(-1, 1)
        self._y_test = self._y[n_of_test:].reshape(-1, 1)

        # 모델 선언
        self._model = BaggingRegressor()

        # 모델 학습
        self._model.fit(self._x_train, self._y_train)

        # 그리드 서치 모델
        # self._g_model = None
        print("================ BaggingRegressor __init__ End =============")

    # 일반 예측
    def predict(self, save_img=False, show_chart=False):
        print("================ BaggingRegressor predict Start =============")

        # r2_score 전처리 데이터 예측
        y_pred = self._model.predict(self._x_test)

        # 스코어 정보
        score = r2_score(self._y_test, y_pred)

        # 리포트 확인
        if hasattr(self._model, 'coef_') and hasattr(self._model, 'intercept_'):
            print(f'Coef = {self._model.coef_}')
            print(f'intercept = {self._model.intercept_}')

        # 예측 값  & 스코어
        print(f'r2_Score = {score}')

        # 이미지 저장 여부
        if save_img:
            self.save_chart_image(y_pred, show_chart)

        print("================ BaggingRegressor predict End =============")
        return score

    # 후처리데이터 생성 및 예측
    def predict_generator(self):
        print("================ BaggingRegressor predict_generator Start =============")
        columns = self.data.columns.tolist()  # 업로드 컬럼 리스트 (후처리시 동일한 컬럼으로 사용)
        maxdate = max(self.data['DATE'])  # 업로드 데이터 최대일자 (후처리시 그 다음날 기준으로 date 생성)

        # 후처리 데이터 생성
        predictgenerator = PredictGeneraotr(name=self._name, learning_column=self.learning_column,
                                            prediction_column=self.prediction_column, columns=columns,
                                            maxdate=maxdate)

        # 학습모델 로드
        model_path = self._f_path + f'/model/{self._name}_rg.pkl'
        load_model = joblib.load(model_path)

        # 후처리데이터 예측
        self._x_test = predictgenerator._x_test.reshape(-1, 1)
        y_pred = load_model.predict(self._x_test)
        print("y_pred: ", y_pred, len(y_pred))

        # 후처리데이터 예측값 추가
        predictgenerator.predict_df[self.prediction_column] = y_pred
        # predict_df_filepath = self._f_path + "/regression/csv/"
        # predict_df_filename = self._name + "_predict.csv"
        # predictgenerator.predict_df.to_csv(predict_df_filepath + predict_df_filename, index=False, mode='w')

        print("================ BaggingRegressor predict_generator End =============")
        return predictgenerator.predict_df

    # 차트 데이터 변환
    def chart_transform(self, chart_info):
        print("================ BaggingRegressor chart_transform Start =============")
        print("chart_info: ", chart_info)

        # x축 컬럼, y축 컬럼, 차트종류
        x = chart_info['x']
        y = chart_info['y']
        type = chart_info['type']

        # 차트에서 사용할 DataFrame
        data = chart_info['data']

        # x컬럼으로 그룹핑하고 y컬럼 값 합침
        data = data.groupby([x])[y].sum().reset_index()
        categories = list(data[x])
        series_name = y
        series_data = round(data[y], 2).tolist()

        # return 차트데이터, 차트옵션
        chart_data = dict(categories=categories, series_name=series_name, series_data=series_data)

        print("================ BaggingRegressor chart_transform End =============")
        return chart_data

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
                          self._f_path + f'/model/{str(self._name) + "_rg_" + str(time.time())}.pkl')
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
    classifier.save_model()

    # 모델 갱신
    # classifier.save_model(renew=True)

    # 그리드 서치 실행
    # classifier.predict_by_gs()
