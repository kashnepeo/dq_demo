import json
import os
import os.path
import time
import pandas as pd
import csv
import flask
from flask import jsonify
from flask_restful import Resource, Api, abort
from flask import Flask, request
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


@app.route("/csvAnalysis")
def csvAnalysis():
    return flask.render_template("csvAnalysis/csvAnalysis.html")


@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file2']
        f.save(secure_filename(f.filename))
        f = open(secure_filename(f.filename))
        lists = csv.reader(f)
        resultList = []
        for list in lists:
            resultList.append([except_fn(x) for x in list])
        f.close
        return jsonify(resultList)


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
        csv_file = request.files['csv_file']
        classifier_algorithm = request.form['classifier_algorithm']
        _f_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
        csv_file.save(os.path.join(_f_path, 'ml_rest', 'ml', 'classifier', 'resource', secure_filename(csv_file.filename)))

        # 분류 객체 생성(Str -> Class)
        try:
            cls = eval(classifier_algorithm + '.' + app.config['algorithm']['classifier'][classifier_algorithm])(filename=csv_file.filename)
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


api.add_resource(ClassifierHandler, '/classifier')
api.add_resource(RegressionHandler, '/regression/<string:element>')
api.add_resource(NltkHandler, '/nltk/<string:element>')

if __name__ == '__main__':
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
