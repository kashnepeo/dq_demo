import csv
import datetime
import json
import math
import os
import os.path
import time
import numpy as np
import flask
import pymysql
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from flask import Flask, request, url_for
from flask_restful import Resource, Api, abort
from ml.classifier import *
from ml.regression import *
from werkzeug.utils import secure_filename
from datetime import datetime

# 환경 정보 로드
with open('./system/config.json', 'rt', encoding='utf-8') as j:
    config = json.loads(j.read())
# 플라스크 객체 선언
app = Flask(__name__)
# Config Update
app.config.update(config)
# 서버 인스턴스
api = Api(app)
# csv 총 업로드 Row 수
csvTotRow = 0

# log 전체를 입력하기 위해
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # or whatever
handler = logging.FileHandler('log/main.log', 'w', 'utf8')
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(lineno)3s:%(funcName)-15s %(message)s'))
root_logger.addHandler(handler)

# terminal logging
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(levelname)-7s %(lineno)3s:%(funcName)-15s %(message)s'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# directory logging
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = current_dir + os.sep + "log" + os.sep + os.path.splitext(
    os.path.basename(__file__)
)[0] + ".log"
handler = TimedRotatingFileHandler(
    filename=filename, when='midnight', backupCount=7, encoding='utf8'
)
handler.suffix = '%Y%m%d'
formatter = logging.Formatter(
    '%(asctime)s %(levelname)-7s %(lineno)3s:%(funcName)-15s %(message)s'
)
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route("/analysis")
def analysis():
    return flask.render_template("analysis/analysis.html")


@app.route("/example")
def example():
    return flask.render_template("exampleUI.html")


@app.route("/verification", methods=['GET'])
def verification():
    model_seq = request.args.get('model_seq')
    return flask.render_template("analysis/verification.html", model_seq=model_seq)


# CSV 업로드
class UploadFile(Resource):
    def post(self):
        if request.method == 'POST':
            file_encoding = request.form['file_encoding']
            # 기본 경로
            self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
            today = datetime.today()
            # 디렉토리 확인
            if not os.path.isdir(
                    self._f_path + '/ml_rest/ml/' + request.form['model_category'] + '/resource/' + today.strftime(
                        '%Y%m%d')):
                os.makedirs(
                    self._f_path + '/ml_rest/ml/' + request.form['model_category'] + '/resource/' + today.strftime(
                        '%Y%m%d'))
            # CSV 업로드
            f = request.files['file2']
            f.save(self._f_path + '/ml_rest/ml/' + request.form['model_category'] + '/resource/' + today.strftime(
                '%Y%m%d') + '/' + secure_filename(f.filename))
            f = open(self._f_path + '/ml_rest/ml/' + request.form['model_category'] + '/resource/' + today.strftime(
                '%Y%m%d') + '/' + secure_filename(f.filename), encoding=file_encoding)
            # CSV 데이터 파싱
            lists = csv.reader(f)
            resultList = []
            for list in lists:
                resultList.append([except_fn(x) for x in list])
            f.close

            if len(resultList) > 5000:
                resultList.clear()

            # 업로드 CSV 데이터
            global csvTotRow
            csvTotRow = len(resultList)

            # 응답 헤더
            response_data = app.response_class(
                response=json.dumps(resultList),
                status=200,
                mimetype='application/json'
            )

        return response_data


class CsvInfoCU(Resource):
    def post(self):
        if request.method == 'POST':
            reg_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            db_class = Database()
            sql = "INSERT INTO dev.csv_dump_table("
            if request.form['model_name'] is not None:
                sql += "model_nm"
            if request.form['model_category'] is not None:
                sql += ", model_category"
            if request.form['model_algorithm'] is not None:
                sql += ", model_algorithm"
            if request.form['model_cd'] is not None:
                sql += ", model_cd"
            if request.form['model_learning'] is not None:
                sql += ", model_learning"
            if request.form['model_prediction'] is not None:
                sql += ", model_prediction"
            if request.form['csv_filename'] is not None:
                sql += ", csv_filename"
            sql += ", reg_date"
            sql += ") VALUES ("
            if request.form['model_name'] is not None:
                sql += "'%s'" % (request.form['model_name'])
            if request.form['model_category'] is not None:
                sql += ", '%s'" % (request.form['model_category'])
            if request.form['model_algorithm'] is not None:
                sql += ", '%s'" % (request.form['model_algorithm'])
            if request.form['model_cd'] is not None:
                sql += ", '%s'" % (request.form['model_cd'])
            if request.form['model_learning'] is not None:
                sql += ", '%s'" % (request.form['model_learning'])
            if request.form['model_prediction'] is not None:
                sql += ", '%s'" % (request.form['model_prediction'])
            if request.form['csv_filename'] is not None:
                sql += ", '%s'" % (request.form['csv_filename'])
            sql += ", '{}'".format(reg_date)
            sql += ")"
            db_class.execute(sql)
            db_class.commit()
            sql = "SELECT last_insert_id() AS upload_csv_seq"
            row = db_class.executeOne(sql)
            # 응답 헤더
            response_data = app.response_class(
                response=json.dumps(row),
                status=200,
                mimetype='application/json'
            )
        return response_data


class SelectGridHandler(Resource):
    def post(self):
        if request.method == 'POST':
            db_class = Database()
            sql = ""
            if request.form['model_category'] == 'F1 score':
                sql = "SELECT class_cd AS '상품유형 코드' , class_cd_nm AS '카테고리 명' , lrn_count AS '트레이닝 건수' , vrfc_count AS '검증 건수' ," \
                      " prec AS 'Precision' , recal AS 'Recall' , fonescore AS 'F1Score' FROM classifier_model_view WHERE model_seq = %s" \
                      % (request.form['model_seq'])
            print(sql)
            # 데이타 Fetch
            row = db_class.executeAll(sql)
            # 응답 헤더
            response_data = app.response_class(
                response=json.dumps(row),
                status=200,
                mimetype='application/json'
            )
        return response_data


def except_fn(x):
    try:
        return "{:d}".format(round(float(x)))
    except ValueError:
        return x


def abort_function():
    abort(404)


# Classifier
class ClassifierHandler(Resource):
    def post(self):
        try:
            csv_file = request.files['fileObj']
            _f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
            csv_file.save(
                os.path.join(_f_path, 'ml_rest', 'ml', 'classifier', 'resource', secure_filename(csv_file.filename)))
            classifier_algorithm = request.form['classifier_algorithm']
            print(classifier_algorithm + '.' + app.config['algorithm']['classifier'][classifier_algorithm])

            # 분류 객체 생성(Str -> Class)
            try:
                cls = eval(classifier_algorithm + '.' + app.config['algorithm']['classifier'][classifier_algorithm])(
                    params=request.form, filename=csv_file.filename)
            except KeyError:
                abort_function()

            # 스코어 리턴, 레포트 정보, 테스트셋 분석결과
            score, report_df, output, feature_data, category_list = cls.predict(app.config['cnslTypeLgcsfCd'])

            # csv 컬럼명 포맷을 지정해야함 문서아이디, 원본 카테고리
            prediction_column = request.form['prediction_column']
            recordkey = output[output.columns.values[0]]
            call_l_class_cd = output[prediction_column]

            print('recordkey', recordkey)
            print('prediction_column', prediction_column)
            print('call_l_class_cd', call_l_class_cd)
            predict = output['PREDICT']
            print(type(predict), predict)

            global csvTotRow
            lrn_count = math.floor(csvTotRow * 0.7)
            vrfc_count = csvTotRow - lrn_count
            db_class = Database()

            temp_class_cd = 0
            for (f, s, t, r, v) in report_df.values:
                sql = "REPLACE INTO dev.classifier_model_view( model_seq, class_cd, class_cd_nm, lrn_count, vrfc_count, prec, recal, fonescore, reg_date ) VALUES ("
                sql += str(request.form['model_seq']) + ","
                if f.isdigit():
                    sql += f + ",'"
                else:
                    sql += str(temp_class_cd) + ",'"
                if f in app.config['cnslTypeLgcsfCd'].keys():
                    sql += app.config['cnslTypeLgcsfCd'][f] + "',"
                else:
                    sql += f + "',"
                sql += str(lrn_count) + ","
                sql += str(vrfc_count) + ",'"
                sql += str(s) + "','"
                sql += str(t) + "','"
                sql += str(r) + "','"
                sql += datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "')"
                temp_class_cd += 1
                print(sql)
                db_class.execute(sql)
                db_class.commit()
            # 응답 데이터
            print('report_df', report_df)
            data = dict(name=classifier_algorithm, category='classifier', success=True, score=score,
                        report_value=report_df['precision'].tolist(), report_lable=report_df['class'].tolist(),
                        cv_score=list(), gs_score=cls.predict_by_gs(), req_time=time.time(),
                        recordkey=recordkey.tolist()  # [:10]
                        , call_l_class_cd=call_l_class_cd.tolist()  # [:10],
                        , predict=predict.tolist()  # [:10]
                        , feature_data=feature_data, category_list=category_list)

            # 모델 Payload 확인
            if os.path.isfile(f'./ml/model/{classifier_algorithm}.pkl'):
                print(f'{classifier_algorithm} Model Exist,')
            else:
                print(f'{classifier_algorithm} Model Not Exist,')
                # 최초 모델 생성
                cls.save_model()

                # 응답 헤더
        except Exception as ex:
            logger.debug(ex)
            abort(500)

        response_data = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )

        return response_data


# Regression
class RegressionHandler(Resource):
    def post(self):
        print("================ RegressionHandler post Start =============")
        print("RegressionHandler post: ", request)
        print("request: ", request.form)
        # print("modelName: ", request.form['modelName'])  # 모델명
        # print("subject: ", request.form['subject'])  # 모델명
        # print("regression_algorithm: ", request.form['regression_algorithm'])  # 알고리즘명
        # print("model_save: ", request.form['model_save'])  # 모델저장 선택 (가망고객분석, 고객반응분석, 이탈고객분석, 민원고객분석)
        # print("learning_column: ", request.form['learning_column'])  # 학습컬럼
        # print("prediction_column: ", request.form['prediction_column'])  # 예측컬럼
        # print("view_chart: ", request.form['view_chart'])  # 차트 (라인차트)
        # print("model_seq: ", request.form['model_seq'])  # 모델시퀀스번호
        # print("fileObj: ", request.files['fileObj'])  # 업로드한 CSV파일정보
        # print("fileObj filename: ", request.files['fileObj'].filename)  # 업로드한 CSV파일정보

        _f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

        regression_algorithm = request.form['regression_algorithm']

        # 1. 업로드한 CSV 서버 디렉토리에 저장
        upload_csv_file = request.files['fileObj']
        filename = regression_algorithm + '_' + upload_csv_file.filename
        upload_csv_file.save(
            os.path.join(_f_path, 'ml_rest', 'ml', 'regression', 'csv',
                         secure_filename(regression_algorithm + '_' + upload_csv_file.filename)))

        # 분류 객체 생성(Str -> Class)
        # 2. 선택한 알고리즘모델 학습 (EX. bagging_rg 실행)
        try:
            cls = eval(regression_algorithm + '_rg.' + app.config['algorithm']['regression'][regression_algorithm])(
                params=request.form, filename=filename)
        except KeyError:
            abort_function()

        # 3. 선택한 알고리즘모델 예측 실행 (r2_score 및 다른 검증방식 CSV 저장)
        r2_score = cls.predict()
        print("r2_score: ", r2_score)

        # 4. 학습모델 저장
        if os.path.isfile(f'./ml/model/{regression_algorithm}_rg.pkl'):
            print(f'{regression_algorithm}_rg Model Exist,')
            cls.save_model(renew=True)
        else:
            print(f'{regression_algorithm}_rg Model Not Exist,')
            # 최초 모델 생성
            cls.save_model()

        # 5. 후처리데이터 생성 및 예측 CSV 저장
        predict_df = cls.predict_generator()
        print("main.py predict_df: ", predict_df.head(5), predict_df.info())

        predict_df.to_csv(os.path.join(_f_path, 'ml_rest', 'ml', 'regression', 'csv',
                                       secure_filename(regression_algorithm + '_predict_' + upload_csv_file.filename)),
                          index=False, mode='w')

        # 6. 선택한 차트로 데이터 구성 (x축: 날짜, y축: 콜 예측인입량 (후처리 CSV)
        chart_info = dict(x=request.form['learning_column'], y=request.form['prediction_column'],
                          type=request.form['view_chart'], data=predict_df)
        chart_data = cls.chart_transform(chart_info)

        print("chart_data: ", chart_data, type(chart_data))

        # 응답 헤더
        response_data = app.response_class(
            response=json.dumps(chart_data),
            status=200,
            mimetype='application/json'
        )

        print("================ RegressionHandler post End =============")
        return response_data

    def get(self, element):

        print("RegressionHandler get: ", element)

        # 분류 객체 생성(Str -> Class)
        try:
            cls = eval(element + '_rg.' + app.config['algorithm']['regression'][element])()
        except KeyError:
            abort_function()

        # 응답 데이터
        data = dict(name=element, category='regression', success=True, pre_data=cls.predict()[0],
                    score=cls.predict()[1], cv_score=cls.predict_by_cv(),
                    gs_score=cls.predict_by_gs(),
                    req_time=time.time())

        # 모델 Payload 확인
        if os.path.isfile(f'./ml/model/{element}.pkl'):
            print(f'{element} Model Exist,')
        else:
            print(f'{element} Model Not Exist,')
            # 최초 모델 생성
            cls.save_model()

        # 응답 헤더
        response_data = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )

        return response_data


# NLTK
class NltkHandler(Resource):
    def get(self, element):
        pass


# DB
class Database():
    def __init__(self):
        self.db = pymysql.connect(host=app.config['dbInfo']['host'],
                                  user=app.config['dbInfo']['user'],
                                  password=app.config['dbInfo']['password'],
                                  db=app.config['dbInfo']['db'],
                                  charset=app.config['dbInfo']['charset'])
        self.cursor = self.db.cursor(pymysql.cursors.DictCursor)

    def execute(self, query, args={}):
        self.cursor.execute(query, args)

    def executeOne(self, query, args={}):
        self.cursor.execute(query, args)
        row = self.cursor.fetchone()
        return row

    def executeAll(self, query, args={}):
        self.cursor.execute(query, args)
        row = self.cursor.fetchall()
        return row

    def commit(self):
        self.db.commit()


api.add_resource(ClassifierHandler, '/classifier')
# api.add_resource(RegressionHandler, '/regression/<string:element>')
api.add_resource(RegressionHandler, '/regression')
api.add_resource(NltkHandler, '/nltk/<string:element>')
api.add_resource(UploadFile, '/fileUpload')
api.add_resource(CsvInfoCU, '/csvInfoCU')
api.add_resource(SelectGridHandler, '/selectGrid')
if __name__ == '__main__':
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=5000, debug=True)
