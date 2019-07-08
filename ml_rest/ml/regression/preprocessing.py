import os
import warnings

import pandas as pd

from sklearn.preprocessing import LabelEncoder
class Preprocessing():

    # 초기 init 함수
    def __init__(self, filepath, filename):
        print(
            "============================================== preprocessing __init__ 전처리 Start ==============================================")
        print("filepath: ", filepath)
        print("filename: ", filename)

        self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
        print("_f_path: ", self._f_path)

        # 경고 메시지 삭제
        warnings.filterwarnings('ignore')

        # 원본 데이터 로드
        self.original_df = pd.read_csv(filepath + filename)

        # 기존 원본컬럼 + 날짜, 시간, 요일 데이터 추가 DataFrame 구성
        self.addcolumn_df = self.date_split()

        # 센터, 콜타입 라벨인코딩 변환
        self.addcolumn_df = self.label_encoding()
        print("addcolumn_df: ", self.addcolumn_df.head(5))

        # 전처리 CSV 생성
        self.preprocess_df = self.write_preprocessing(self._f_path)

        # 원본df: self.original_df
        # 추가df: self.addcolumn_df
        # 전처리df: self.preprocess_df

        # 학습 및 레이블(정답) 데이터 분리
        self._x = self.preprocess_df.iloc[:, :-1].values
        self._y = self.preprocess_df.iloc[:, -1].values

        print(
            "============================================== preprocessing __init__ 전처리 End ==============================================")

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
        print(
            "============================================== date_split 날짜, 시간, 요일, 콜빈도수 함수 Start ==============================================")

        # 1. original.csv STT_CMDTM 사용
        stt_cmdtm = self.original_df['STT_CMDTM']

        # 2. 날짜(yyyy-MM-dd) 데이터 분리
        date = stt_cmdtm.str.replace('-', '').str[:8].astype('int')

        # 3. 데이터 시간(hh) 데이터 분리
        hour = stt_cmdtm.str[11:13].str.replace(':', '').astype('int')

        # 4. 요일 구하기
        # day_name = pd.to_datetime(date).dt.day_name()  # 요일 이름
        dayofweek = pd.to_datetime(date).dt.dayofweek  # 요일 숫자
        # 월:0 , 화:1 , 수:2 , 목:3 , 금:4 , 토:5 , 일:6

        # 4. 날짜, 시간, 요일 데이터 return
        addcolumn_df = pd.DataFrame(
            {'RECORDKEY': self.original_df['RECORDKEY'], 'DATE': date, 'HOUR': hour,
             'DAYOFWEEK': dayofweek})

        print(
            "============================================== date_split 날짜, 시간, 요일, 콜빈도수 함수 End ==============================================")

        return addcolumn_df

    def label_encoding(self):
        print(
            "============================================== label_encoding 센터, 콜타입 함수 Start ==============================================")

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

        print(
            "============================================== label_encoding 센터, 콜타입 함수 End ==============================================")

        return self.addcolumn_df

    # 전처리 csv 생성
    def write_preprocessing(self, filepath):
        print(
            "============================================== write_preprocessing 전처리 csv생성 Start ==============================================")
        # pre = self.original_df.append(self.preprocess_df)

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
        self.preprocess_filename = "preprocessing.csv"
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        preprocess_df.to_csv(self.preprocess_filepath + self.preprocess_filename, index=False, mode='w')

        print(
            "============================================== write_preprocessing 전처리 csv생성 End ==============================================")

        return preprocess_df