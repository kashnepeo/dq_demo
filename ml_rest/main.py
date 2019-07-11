import csv
import json
import os
import os.path
import time
import pymysql
import flask
import math
import datetime
import pandas as pd
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, abort
from ml.classifier import *
from ml.regression import *
from werkzeug.utils import secure_filename

# 환경 정보 로드
with open('./system/config.json', 'rt', encoding='utf-8') as j:
    config = json.loads(j.read())

# 플라스크 객체 선언
app = Flask(__name__)

# Config Update
app.config.update(config)

# 서버 인스턴스
api = Api(app)

csvTotRow = 0


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
            # 기본 경로
            self._f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
            today = datetime.datetime.today()
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
                '%Y%m%d') + '/' + secure_filename(f.filename))
            # CSV 데이터 파싱
            arr = []
            with f as csvFile:
                csvReader = csv.DictReader(csvFile)
                for csvRow in csvReader:
                    arr.append(csvRow)
            # 업로드 CSV 데이터
            global csvTotRow
            csvTotRow = len(arr)

            # 응답 헤더
            response_data = app.response_class(
                response=json.dumps(arr),
                status=200,
                mimetype='application/json'
            )

        return response_data


class CsvInfoCU(Resource):
    def post(self):
        if request.method == 'POST':
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
        score, report_df, output, feature_data = cls.predict(app.config['cnslTypeLgcsfCd'])

        recordkey = output['recordkey']
        call_l_class_cd = output['call_l_class_cd']
        predict = output['predict']
        print(type(predict))

        global csvTotRow
        lrn_count = math.floor(csvTotRow * 0.7)
        vrfc_count = csvTotRow - lrn_count
        db_class = Database()
        for (f, s, t, r, v) in report_df.values:
            sql = "INSERT INTO dev.classifier_model_view( model_seq, class_cd, class_cd_nm, lrn_count, vrfc_count, prec, recal, fonescore ) VALUES ("
            sql += str(request.form['model_seq']) + ","
            sql += f + ",'"
            sql += app.config['cnslTypeLgcsfCd'][f] + "',"
            sql += str(lrn_count) + ","
            sql += str(vrfc_count) + ",'"
            sql += str(s) + "','"
            sql += str(t) + "','"
            sql += str(r) + "')"
            print(sql)
            db_class.execute(sql)
            db_class.commit()
        # 응답 데이터
        data = dict(name=classifier_algorithm, category='classifier', success=True, score=score,
                    report_value=report_df['precision'].tolist(), report_lable=report_df['class'].tolist(),
                    cv_score=list(), gs_score=cls.predict_by_gs(), req_time=time.time(),
                    recordkey=recordkey.tolist()[:10], call_l_class_cd=call_l_class_cd.tolist()[:10],
                    predict=predict.tolist()[:10], feature_data=feature_data)

        # 모델 Payload 확인
        if os.path.isfile(f'./ml/model/{classifier_algorithm}.pkl'):
            print(f'{classifier_algorithm} Model Exist,')
        else:
            print(f'{classifier_algorithm} Model Not Exist,')
            # 최초 모델 생성
            cls.save_model()

        # 응답 헤더
        response_data = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )

        return response_data


# Regression
class RegressionHandler(Resource):
    def post(self):
        print("RegressionHandler post: ", request)


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
    app.run(host='0.0.0.0', port=8000, debug=True)
