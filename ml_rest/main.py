from flask import Flask
from flask_restful import Resource, Api
from sklearn.externals import joblib
from ml.classifier import *


app = Flask(__name__)
api = Api(app)


# classifier
class Classifier(Resource):
    def get(self, element):
        if element == 'AdaBoostClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'BeggingClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'BernoulliNB':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'CalibratedClassifierCV':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'ComplementNB':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'DecisionTreeClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'ExtraTreeClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'ExtraTreesClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'GaussianNB':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'GaussianProcessClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'GradientBoostingClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'KNeighborsClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'LabelPropagation':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'LabelSpreading':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'LinearDiscriminantAnalysis':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'LinearSVC':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'LogisticRegression':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'LogisticRegressionCV':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'MLPClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'MultinomialNB':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'NearestCentroid':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'NuSVC':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'PassiveAggresiveClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'Perceptron':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'QuadraticDisriminantAnalysis':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'RadiusNeighborsClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'RandomForestClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'RidgeClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'RidgeClassifierCV':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'SGDClassifier':
            ada = AdaBoostClass()
            return ada.predict()
        elif element == 'SVC':
            ada = AdaBoostClass()
            return ada.predict()


# regression
class Regression(Resource):
    def get(self, element):
        if element == 'ARDRegression':
            return {'name': 'ARDRegression', 'result': False}
        elif element == 'AdaBoostRegressor':
            return {'name': 'AdaBoostRegressor', 'result': False}
        elif element == 'BaggingRegressor':
            return {'name': 'BaggingRegressor', 'result': False}
        elif element == 'ByesianRidge':
            return {'name': 'ByesianRidge', 'result': False}
        elif element == 'CCA':
            return {'name': 'CCA', 'result': False}
        elif element == 'DecisionTreeRegressor':
            return {'name': 'DecisionTreeRegressor', 'result': False}
        elif element == 'ElasticNet':
            return {'name': 'ElasticNet', 'result': False}
        elif element == 'ElasticNetCV':
            return {'name': 'ElasticNetCV', 'result': False}
        elif element == 'ExtraTreeRegressor':
            return {'name': 'ExtraTreeRegressor', 'result': False}
        elif element == 'ExtraTreesRegressor':
            return {'name': 'ExtraTreesRegressor', 'result': False}
        elif element == 'GaussianProcessRegressor':
            return {'name': 'GaussianProcessRegressor', 'result': False}
        elif element == 'GradientBoostingRegressor':
            return {'name': 'GradientBoostingRegressor', 'result': False}
        elif element == 'HuberRegressor':
            return {'name': 'HuberRegressor', 'result': False}
        elif element == 'KNeighborsRegressor':
            return {'name': 'KNeighborsRegressor', 'result': False}
        elif element == 'KernelRidge':
            return {'name': 'KernelRidge', 'result': False}
        elif element == 'Lars':
            return {'name': 'Lars', 'result': False}
        elif element == 'LarsCV':
            return {'name': 'LarsCV', 'result': False}
        elif element == 'Lasso':
            return {'name': 'Lasso', 'result': False}
        elif element == 'LassoCV':
            return {'name': 'LassoCV', 'result': False}
        elif element == 'LassoLars':
            return {'name': 'LassoLars', 'result': False}
        elif element == 'LassoLarsCV':
            return {'name': 'LassoLarsCV', 'result': False}
        elif element == 'LassoLarslC':
            return {'name': 'LassoLarslC', 'result': False}
        elif element == 'LinearRegression':
            return {'name': 'LinearRegression', 'result': False}
        elif element == 'LinearSVR':
            return {'name': 'LinearSVR', 'result': False}
        elif element == 'MLPRegressor':
            return {'name': 'MLPRegressor', 'result': False}
        elif element == 'NuSVR':
            return {'name': 'NuSVR', 'result': False}
        elif element == 'LinearSVR':
            return {'name': 'LinearSVR', 'result': False}
        elif element == 'MLPRegressor':
            return {'name': 'MLPRegressor', 'result': False}
        elif element == 'NuSVR':
            return {'name': 'NuSVR', 'result': False}
        elif element == 'OrthogonalMatchingPursuit':
            return {'name': 'OrthogonalMatchingPursuit', 'result': False}
        elif element == 'OrthogonalMatchingPursuitCV':
            return {'name': 'OrthogonalMatchingPursuitCV', 'result': False}
        elif element == 'PLSCanonical':
            return {'name': 'PLSCanonical', 'result': False}
        elif element == 'PLSRegression':
            return {'name': 'PLSRegression', 'result': False}
        elif element == 'PassiveAggressiveRegressor':
            return {'name': 'PassiveAggressiveRegressor', 'result': False}
        elif element == 'RandomForestRegressor':
            return {'name': 'RandomForestRegressor', 'result': False}
        elif element == 'Ridge':
            return {'name': 'Ridge', 'result': False}
        elif element == 'RidgeCV':
            return {'name': 'RidgeCV', 'result': False}
        elif element == 'SGDRegressor':
            return {'name': 'SGDRegressor', 'result': False}
        elif element == 'SVR':
            return {'name': 'SVR', 'result': False}
        elif element == 'TheilSenRegressor':
            return {'name': 'TheilSenRegressor', 'result': False}
        elif element == 'TransformedTargetRegressor':
            return {'name': 'TransformedTargetRegressor', 'result': False}


api.add_resource(Classifier, '/classifier/<string:element>')
api.add_resource(Regression, '/regression/<string:element>')

if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    model = joblib.load('./model/model.pkl')

    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
