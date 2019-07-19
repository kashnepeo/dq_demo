import os
import warnings

import pandas as pd

from sklearn.preprocessing import LabelEncoder


class Preprocessing():

    # 초기 init 함수
    def __init__(self, name, filepath, filename):
        print("================ Regression Preprocessing __init__ Start =============")
        self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')

        # 원본 데이터 로드
        self.original_df = pd.read_csv(filepath + filename)

        # 기존 원본컬럼 + 날짜, 시간, 요일 데이터 추가 DataFrame 구성
        self.addcolumn_df = self.date_split()

        # 센터, 콜타입 라벨인코딩 변환
        self.addcolumn_df = self.label_encoding()

        # 전처리 CSV 생성
        self.preprocess_df = self.write_preprocessing(name, self._f_path)

        # 원본df: self.original_df
        # 추가df: self.addcolumn_df
        # 전처리df: self.preprocess_df

        print("================ Regression Preprocessing __init__ End =============")

    # 날짜, 시간, 요일, 콜빈도수 구하는 함수
    """
    original.csv 로드해서 STT_CMDTM(STT완료일시)를 1시간 기준으로 분할 및 요일 추가(컬럼추가: DATE, HOUR, DAYOFWEEK)
    [original_name]_preprocessing_[regressor_name].csv 생성
    EX. 
	RECORDKEY 	   STT_CMDTM            CENTER		 CALL_TYPE		 DATE 		HOUR 	DAYOFWEEK	CALL_TOTAL
	1234         2019-01-01 1:23         DFS           OB          2019-01-01    1          2			1
	5678         2019-01-01 1:56         DFS           IB          2019-01-01    1          2			1
    """

    def date_split(self):
        print("================ Regression Preprocessing date_split Start =============")

        # 1. original.csv STT_CMDTM 사용
        stt_cmdtm = self.original_df['STT_CMDTM']

        # 2. 날짜(yyyy-MM-dd) 데이터 분리
        date = stt_cmdtm.str.replace('-', '').str[:8].astype('int')

        # 3. 데이터 시간(hh) 데이터 분리
        hour = stt_cmdtm.str[11:13].str.replace(':', '').astype('int')
        # 4. 요일 구하기
        tempdate = stt_cmdtm.str[:10]
        # day_name = pd.to_datetime(tempdate).dt.day_name()  # 요일 이름
        dayofweek = pd.to_datetime(tempdate).dt.dayofweek  # 요일 숫자
        # 월:0 , 화:1 , 수:2 , 목:3 , 금:4 , 토:5 , 일:6

        # 4. 날짜, 시간, 요일 데이터 return
        addcolumn_df = pd.DataFrame(
            {'RECORDKEY': self.original_df['RECORDKEY'], 'DATE': date, 'HOUR': hour,
             'DAYOFWEEK': dayofweek})

        print("================ Regression Preprocessing date_split End =============")
        return addcolumn_df

    def label_encoding(self):
        print("================ Regression Preprocessing label_encoding Start =============")

        le = LabelEncoder()

        # 센터 라벨인코딩
        center = le.fit_transform(self.original_df['CENTER'])
        if max(center) != 0:
            center = center / max(center)
        self.addcolumn_df['CENTER_LE'] = center.astype('int')

        # 콜타입 라벨인코딩
        call_type = le.fit_transform(self.original_df['CALL_TYPE'])
        if max(call_type) != 0:
            call_type = call_type / max(call_type)
        self.addcolumn_df['CALL_TYPE_LE'] = call_type.astype('int')

        print("================ Regression Preprocessing label_encoding End =============")
        return self.addcolumn_df

    # 전처리 csv 생성
    def write_preprocessing(self, name, filepath):
        print("================ Regression Preprocessing write_preprocessing Start =============")

        # 원본df, 추가df merge. 키값이 있어서 merge가능(key: RECORDKEY)
        preprocess_df = self.original_df.merge(self.addcolumn_df)

        # merge후 키 제거
        del self.addcolumn_df['RECORDKEY']

        # 모든 콜 빈도수 1로 지정
        preprocess_df['CALL_TOTAL'] = 1

        # 날짜, 시간, 요일, 센터, 콜타입으로 그룹핑하고 콜빈도수 계산
        preprocess_df = preprocess_df.groupby(['DATE', 'HOUR', 'DAYOFWEEK', 'CENTER_LE', 'CALL_TYPE_LE'])[
            'CALL_TOTAL'].sum().reset_index()

        # 전처리 csv 생성
        self.preprocess_filepath = filepath + "/regression/csv/"
        self.preprocess_filename = name + "_preprocessing.csv"
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        preprocess_df.to_csv(self.preprocess_filepath + self.preprocess_filename, index=False, mode='w')

        print("================ Regression Preprocessing write_preprocessing End =============")
        return preprocess_df
