import os
import time
import warnings

import pandas as pd
from sklearn.externals import joblib
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# from ml_rest.ml.classifier.preprocessing import Preprocessing
from .preprocessing import Preprocessing


class GaussianProcessClass:
    """
    Name      : GaussianProcessClassifier
    Attribute : None
    Method    : predict, predict_by_cv, save_model
    """

    def __init__(self, params, filename):
        # 알고리즘 이름
        self._name = 'gaussianprocess'
        # 기본 경로
        self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')

        subject = params['subject']
        classifier_algorithm = params['classifier_algorithm']
        model_save = params['model_save']
        learning_column = params['learning_column']
        prediction_column = params['prediction_column']
        file_encoding = params["file_encoding"]

        print(model_save, subject, learning_column, prediction_column)

        # 전처리 클래스 생성
        preprocessor = Preprocessing(filename=filename, learning=learning_column, prediction=prediction_column, encoding=file_encoding)

        # 학습 및 레이블(정답) 데이터 분리
        self._x = preprocessor._x
        self._y = preprocessor._y
        self.data = preprocessor.df.sample(frac=1).reset_index(drop=True)

        row_count = self.data.shape[0]
        train_size = int(self.data.shape[0] * 0.66)
        test_size = row_count - train_size

        train_set = self.data[:train_size]
        self.test_set = self.data[test_size:]

        # 전체 테스트용
        # train_set = self.data
        # self.test_set = self.data

        self._x_train = train_set[learning_column].values.astype('U')
        self._y_train = train_set[prediction_column].values.astype('U')
        self._x_test = self.test_set[learning_column].values.astype('U')
        self._y_test = self.test_set[prediction_column].values.astype('U')

        # 전처리 데이터 로드
        self.X_train_tfidf_vector, self.X_test_tfidf_vector, self.vocab, self.dist, self.result_a = preprocessor.keyword_vectorizer(x_train=self._x_train, x_test=self._x_test)

        # 모델 선언
        self._model = GaussianProcessClassifier()

        # 전처리 데이터를 이용한 모델 학습
        self._model.fit(self.X_train_tfidf_vector, self._y_train)

        # 그리드 서치 모델
        self._g_model = None
        self.output = None
        self.y_pred = None

    # 일반 예측
    def predict(self, config):
        # 예측
        self.y_pred = self._model.predict(self.X_test_tfidf_vector)

        # 리포트 출력
        report_str = classification_report(self._y_test, self.y_pred)
        # print(report_str)

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

        feature_data, category_list = self.make_feature_data(config)
        self.test_set['PREDICT'] = self.y_pred
        self.test_set['KEYWORDS'] = self.result_a

        # 분류 결과 file write
        self.test_set.to_csv(self._f_path + f'/classifier/csv/result_{self._name}.csv', index=False, quoting=3, escapechar='\\')

        # 분석 feature file write
        pd.DataFrame(self.dist, columns=['None'] if len(self.vocab) == 1 and self.vocab[0] == '' else self.vocab).to_csv(self._f_path + f'/classifier/csv/features_{self._name}.csv', index=False, quoting=3)

        # 스코어 리턴, 레포트 정보, 테스트셋 분석결과
        return score, report_df, self.test_set, feature_data, category_list

    #  CV 예측(Cross Validation)
    def predict_by_cv(self):
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
        del self._x_train, self._x_test, self._y_train, self._y_test, self._x, self._y, self._model, self.X_train_tfidf_vector, self.X_test_tfidf_vector, self.vocab, self.dist, self.test_set

    def make_feature_data(self, config):
        cat_dict = {}
        for vect, pred in zip(self.result_a, self.y_pred):
            # print(pred, vect)
            if pred in cat_dict.keys():
                cat_dict[pred].update(vect)
            else:
                cat_dict[pred] = vect
        # print(cat_dict)
        for key, value in cat_dict.items():
            cat_dict[key] = sorted(value.items(), key=(lambda x: x[1]), reverse=True)[:20]

        category_list = [config[str(key)] if str(key) in config.keys() else key for key in cat_dict.keys()]

        # print(cat_dict)
        feature_data = []
        for key, value in cat_dict.items():
            for each_word in value:
                feature_data.append(
                    {"x": str(each_word[0]), "value": float(each_word[1]), "category": config[str(key)].replace(' ', '\n').replace('/', '\n') if str(key) in config.keys() else key.replace(' ', '\n').replace('/', '\n')}
                )
        # print(feature_data)
        return feature_data, category_list


if __name__ == "__main__":
    # 클래스 선언
    classifier = GaussianProcessClass({'subject': 'a', 'classifier_algorithm': 'c', 'model_save': 'aaa', 'learning_column': 'TALK', 'prediction_column': 'NICKNAME'}, 'result_3000.csv')

    config = {}
    config['a'] = 1
    # 분류 실행
    classifier.predict(config)

    # 분류 실행(Cross Validation)
    # classifier.predict_by_cv()

    # 모델 신규
    classifier.save_model()

    # 모델 갱신
    # classifier.save_model(renew=True)

    # 그리드 서치 실행
    # classifier.predict_by_gs()
