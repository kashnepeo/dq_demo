import os
import warnings

import numpy as np
import pandas as pd

from datetime import timedelta, datetime

# from ml_rest.ml.regression.preprocessing import *
from .preprocessing import *


class PredictGeneraotr():

    # 초기 init 함수
    def __init__(self, name, learning_column, prediction_column, columns, maxdate):
        print("================ Regression PredictGeneraotr __init__ Start =============")
        self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')

        # 후처리 DataFrame 생성
        self.predict_df = self.date_generator(learning_column, prediction_column, columns, maxdate, days=30)

        # 후처리 CSV 생성
        # self.write_predict(name, self._f_path)
        # print("predict_df: ", self.predict_df.head(5))
        # self.predict_df.to_csv(self._f_path + "/regression/csv/" + name + "_temp.csv", index=False, mode='w')

        # 학습 및 레이블(정답) 데이터 분리
        self._x_test = self.predict_df.loc[:, learning_column].values

        print("================ Regression PredictGeneraotr __init__ End =============")

    # 전처리데이터 마지막날짜 기준 30일 까지 후처리DataFrame 생성
    def date_generator(self, learning_column, prediction_column, columns, maxdate, days=30):
        print("================ Regression PredictGeneraotr date_generator Start =============")

        year = str(maxdate)[:4]
        month = str(maxdate)[4:6]
        day = str(maxdate)[6:8]
        maxdate = datetime.strptime(year + "-" + month + "-" + day, '%Y-%m-%d').date()  # 전처리 데이터 마지막날짜

        columns.remove(prediction_column)  # 정답(label) 컬럼 삭제

        # 학습컬럼이 들어올 때 컬럼의 속성을 알 수 없어서 어떻게 처리해야할지 고민필요...
        temp_list = []
        for day in range(days):  # days
            for hour in range(24):  # hour
                for center_le in range(1):  # center
                    for call_type_le in range(2):  # call_type
                        date_temp = maxdate + timedelta(days=day + 1)
                        dayofweek = date_temp.weekday()
                        date_temp = int(str(date_temp).replace('-', ''))
                        # print("date: ", date_temp, "hour: ", hour, "dayofweek: ", dayofweek, "center_le: ", center_le, "call_type_le: ", call_type_le)
                        temp_list.append([date_temp, hour, dayofweek, center_le, call_type_le])

        predict_df = pd.DataFrame(temp_list, columns=columns)

        print("================ Regression PredictGeneraotr date_generator End =============")
        return predict_df

    # 후처리DataFrame csv 저장
    def write_predict(self, name, filepath):
        print("================ Regression PredictGeneraotr write_predict Start =============")

        # 모든 콜 빈도수 1로 지정
        # predict_df = self.predict_df['CALL_TOTAL'] = 1

        # 후처리 csv 생성
        self.predict_df_filepath = filepath + "/regression/csv/"
        self.predict_df_filename = name + "_predict.csv"
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        self.predict_df.to_csv(self.predict_df_filepath + self.predict_df_filename, index=False, mode='w')

        print("================ Regression PredictGeneraotr write_predict End =============")
