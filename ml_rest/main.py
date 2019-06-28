import csv
import json
import os
import os.path
import time
import pymysql
import flask
import pandas as pd
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, abort
from ml.classifier import *
from ml.regression import *
from werkzeug.utils import secure_filename

# 환경 정보 로드
with open('./system/config.json') as j:
    config = json.loads(j.read())

# 플라스크 객체 선언
app = Flask(__name__)

# Config Update
app.config.update(config)

# 서버 인스턴스
api = Api(app)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route("/analysis")
def analysis():
    return flask.render_template("analysis/analysis.html")


# CSV 업로드
class UploadFile(Resource):
    def post(self):
        if request.method == 'POST':
            f = request.files['file2']
            f.save(secure_filename(f.filename))
            f = open(secure_filename(f.filename))
            lists = csv.reader(f)
            resultList = []
            for list in lists:
                resultList.append([except_fn(x) for x in list])
            f.close

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
            sql = "SELECT last_insert_id()"
            row = db_class.executeOne(sql)
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
        csv_file.save(os.path.join(_f_path, 'ml_rest', 'ml', 'classifier', 'resource', secure_filename(csv_file.filename)))
        classifier_algorithm = request.form['classifier_algorithm']
        print(classifier_algorithm + '.' + app.config['algorithm']['classifier'][classifier_algorithm])

        # 분류 객체 생성(Str -> Class)
        try:
            cls = eval(classifier_algorithm + '.' + app.config['algorithm']['classifier'][classifier_algorithm])(params=request.form, filename=csv_file.filename)
        except KeyError:
            abort_function()

        # 스코어 리턴, 레포트 정보, 테스트셋 분석결과
        score, report_df, output = cls.predict()

        recordkey = output['recordkey']
        call_l_class_cd = output['call_l_class_cd']
        predict = output['predict']
        print(type(predict))
        # 응답 데이터
        data = dict(name=classifier_algorithm, category='classifier', success=True, score=score, report_value=report_df['precision'].tolist(), report_lable=report_df['class'].tolist(),
                    cv_score=list(), gs_score=cls.predict_by_gs(), req_time=time.time(), recordkey=recordkey.tolist()[:10], call_l_class_cd=call_l_class_cd.tolist()[:10], predict=predict.tolist()[:10])

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
    def get(self, element):

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


class Database():
    def __init__(self):
        self.db = pymysql.connect(host='tasvc.diquest.com',
                                  user='diquest',
                                  password='ek2znptm2',
                                  db='dev',
                                  charset='utf8')
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
api.add_resource(RegressionHandler, '/regression/<string:element>')
api.add_resource(NltkHandler, '/nltk/<string:element>')
api.add_resource(UploadFile, '/fileUpload')
api.add_resource(CsvInfoCU, '/csvInfoCU')


if __name__ == '__main__':
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
