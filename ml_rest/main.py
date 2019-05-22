import json
import os
import os.path
import time

import flask
from flask import Flask
from flask_restful import Resource, Api
from ml.classifier import *

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
    return flask.render_template('rest_test.html')


# Classifier
class ClassifierHandler(Resource):
    def get(self, element):

        # 예외 처리(403, 404)
        # 공통 함수로 처리
        # 어재현
        # ****

        # 분류 객체 생성(Str -> Class)
        cls = eval(element + '.' + app.config['algorithm']['classifier'][element])()

        # 응답 데이터
        data = dict(success=True, score=cls.predict(), req_time=time.time())
        print(data)
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
        pass


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
