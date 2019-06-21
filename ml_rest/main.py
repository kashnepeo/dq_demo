import json
import os
import os.path
import time

import flask
from flask import Flask
from flask_restful import Resource, Api, abort
from ml.classifier import *
from ml.regression import *

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


def abort_function():
    abort(404)


# Classifier
class ClassifierHandler(Resource):
    def get(self, element):

        # 분류 객체 생성(Str -> Class)
        try:
            cls = eval(element + '.' + app.config['algorithm']['classifier'][element])()
        except KeyError:
            abort_function()

        # 응답 데이터
        data = dict(name=element, category='classifier', success=True, score=cls.predict(),
                    cv_score=list(cls.predict_by_cv()), gs_score=cls.predict_by_gs(), req_time=time.time())

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


api.add_resource(ClassifierHandler, '/classifier/<string:element>')
api.add_resource(RegressionHandler, '/regression/<string:element>')
api.add_resource(NltkHandler, '/nltk/<string:element>')

if __name__ == '__main__':
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
