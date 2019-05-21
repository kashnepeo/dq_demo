from flask import Flask
from flask_restful import Resource, Api
from sklearn.externals import joblib
from ml.classifier import *

app = Flask(__name__)
api = Api(app)


# Classifier
class ClassifierHandler(Resource):
    def get(self, element):
        pass


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
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    model = joblib.load('./model/model.pkl')

    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
