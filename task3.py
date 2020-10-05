
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
#----------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
from sklearn import datasets

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier

iris=datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target
plt.figure(figsize=(10, 6))
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color='g', label='0')
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='y', label='1')
#plt.show()

norm = preprocessing.normalize(X)
#X = norm
xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size=0.4)

print("Logistic Regression")
logistRegres = LogisticRegression()
logistRegres.fit(xTrain, yTrain)
logistRegresPredict = logistRegres.predict(xTest)
print( classification_report(yTest, logistRegresPredict))
print(accuracy_score(yTest, logistRegresPredict))

print("SVM Model ")
svmModel = SVC(kernel='linear')
svmModel.fit(xTrain, yTrain)
svmModelPredict = svmModel.predict(xTest)
print(classification_report(yTest, svmModelPredict))
print(accuracy_score(yTest, svmModelPredict))

print("kernelSVM Model")
kernelModel = SVC(kernel='rbf')
kernelModel.fit(xTrain, yTrain)
kernelModelPredict = kernelModel.predict(xTest)
print(classification_report(yTest, kernelModelPredict))
print(accuracy_score(yTest, kernelModelPredict))

print("Neural Network")
neurMod = MLPClassifier(solver='lbfgs',max_iter=1000)
neurMod.fit(xTrain, yTrain)
neurModPredict = neurMod.predict(xTest)
print(classification_report(yTest, neurModPredict))
print(accuracy_score(yTest, neurModPredict))
