from flask import Flask
from flask_restful import Resource, Api
import os.path
from ml.classifier import *
import os
import json

# 환경 정보 로드
with open('./system/config.json') as j:
    config = json.loads(j.read())

# 플라스크 객체 선언
app = Flask(__name__)

# Config Update
app.config.update(config)

# 서버 인스턴스
api = Api(app)


# Classifier
class ClassifierHandler(Resource):
    def get(self, element):

        # 예외 처리(403, 404)
        # 공통 함수로 처리
        # 어재현
        # ****

        # 분류 객체 생성(Str -> Class)
        cls = eval(element + '.' + app.config['algorithm']['classifier'][element])()
        # 스코어
        score = cls.predict()
        # Cross Validation 스코어
        cv_score = cls.predict_by_cv()

        # 모델 Payload 확인
        if os.path.isfile(f'./ml/model/{element}.pkl'):
            print(f'{element} Model Exist,')
        else:
            print(f'{element} Model Not Exist,')
            # 최초 모델 생성
            cls.save_model()


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
