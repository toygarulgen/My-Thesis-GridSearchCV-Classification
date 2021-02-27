import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

vowel_train = pd.read_csv('PART2TrainNew.csv', sep=',', header=0)
vowel_test = pd.read_csv('PART2TestNew.csv', sep=',', header=0)
df=vowel_train.append(vowel_test)
df.head()

labels = df.iloc[:,0]
df.drop('Labels', axis=1, inplace=True)

#%% Zero mean unit variance

scaler = preprocessing.StandardScaler().fit(df)
X_scaled = scaler.transform(df)
X_scaled.mean(axis=0)
X_scaled.std(axis=0)

# normed2 = (df - df.mean(axis=0)) / df.std(axis=0)

#%% Splitting Train and test set

X_tr, X_test, y_tr, y_test = train_test_split(X_scaled, labels, test_size = 0.20, shuffle=False)

print ("Train_x Shape: ", X_tr.shape)
print ("Train_y Shape: ", y_tr.shape)
print ("Test_x Shape: ", X_test.shape)
print ("Test_y Shape: ", y_test.shape)
#%% k-NN Algorithm

grid_params = {
    'n_neighbors': np.arange(1, 40), 
    'weights':['uniform','distance'],
    'metric':['euclidean','manhattan','minkowski']
    }

knn = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = grid_params, verbose=1, cv=LeaveOneOut(), n_jobs=-1)
knn_results = knn.fit(X_tr, y_tr)

print('Accuracy of kNN Training: %.3f' % knn_results.best_score_)
print('The best parameters of kNN', knn_results.best_params_)

knnprediction = knn_results.best_estimator_.predict(X_test)
print("Accuracy of kNN Testing:",metrics.accuracy_score(y_test, knnprediction))


#%% Gradient Boosting Algorithm

parameters = {
    "learning_rate":[0.0001, 0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    'n_estimators':[1, 2, 4, 8, 16, 32, 64, 100, 150, 200, 250, 300],
    }

gbc = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = parameters, verbose = 1, cv=LeaveOneOut(), n_jobs=-1)
gbc_results = gbc.fit(X_tr, y_tr)

print('Accuracy of Gradient Boosting Algorithm Training: %.3f' % gbc_results.best_score_)
print('The best parameters of Gradient Boosting Algorithm:', gbc_results.best_params_)

gbcprediction = gbc_results.best_estimator_.predict(X_test)
print("Accuracy of Gradient Boosting Algorithm Testing:",metrics.accuracy_score(y_test, gbcprediction))


#%% AdaBoost Algorithm

search_grid={
    'n_estimators':[1, 2, 4, 8, 16, 32, 64, 100, 150, 200, 250, 300],
    'learning_rate':[0.0001, 0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    }

ada = GridSearchCV(estimator = AdaBoostClassifier(), param_grid = search_grid, verbose = 1, cv=LeaveOneOut(), n_jobs=-1)
ada_results = ada.fit(X_tr, y_tr)

print('Accuracy of AdaBoost Algorithm Training: %.3f' % ada_results.best_score_)
print('The best parameters of AdaBoost Algorithm:', ada_results.best_params_)

adaprediction = ada_results.best_estimator_.predict(X_test)
print("Accuracy of AdaBoost Algorithm:",metrics.accuracy_score(y_test, adaprediction))


#%% Stochastic Gradient Descent Algorithm

grid = {
        'loss' : ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty' : ['l1', 'l2', 'elasticnet'],
        'alpha' : [0.0001, 0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
}

sgd = GridSearchCV(estimator = SGDClassifier(), param_grid = grid, verbose = 1, cv=LeaveOneOut(), n_jobs=-1)
sgd_results = sgd.fit(X_tr, y_tr)

print('Accuracy of Stochastic Gradient Descent Training: %.3f' % sgd_results.best_score_)
print('The best parameters of Stochastic Gradient Descent:', sgd_results.best_params_)

sgdprediction = sgd_results.best_estimator_.predict(X_test)
print("Accuracy of Stochastic Gradient Descent Testing:",metrics.accuracy_score(y_test, sgdprediction))

#%% SVM Algorithm

param_grid = {
    'C' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [0.0001, 0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], 
    'kernel' : ['rbf']
    } 

svm = GridSearchCV(estimator = SVC(), param_grid = param_grid, verbose = 1, cv=LeaveOneOut(), n_jobs=-1)
svm_results = svm.fit(X_tr, y_tr)

print('Accuracy of Support Vector Machine Training: %.3f' % svm_results.best_score_)
print('The best parameters of Support Vector Machine:', svm_results.best_params_)

svmprediction = svm_results.best_estimator_.predict(X_test)
print("Accuracy of Support Vector Machine Testing:",metrics.accuracy_score(y_test, svmprediction))


#%% Decision Tree Classifier

param_grid = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(1,150),
    'min_samples_split' : range(1,10),
    'min_samples_leaf' : range(1,10)
}

dt = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = param_grid, verbose=1, cv=LeaveOneOut(), n_jobs=-1)
dt_results = dt.fit(X_tr, y_tr)

print('Accuracy of Decision Tree Training: %.3f' % dt_results.best_score_)
print('The best parameters of Decision Tree:', dt_results.best_params_)

dtprediction = dt_results.best_estimator_.predict(X_test)
print("Accuracy of Decision Tree Testing:",metrics.accuracy_score(y_test, dtprediction))

#%% Random Forest Classifier
grid_param = { 
    'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 150, 200, 250, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [2,4,6,8],
    'criterion' :['gini', 'entropy']
}

# grid_param = {
#     'max_depth' : range(1,150),
#     'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 150, 200, 250, 300, 350, 400, 500],
#     'bootstrap': [True, False]
# }

rf = GridSearchCV(estimator = RandomForestClassifier(), param_grid = grid_param, verbose=1, cv=LeaveOneOut(), n_jobs=-1)
rf_results = rf.fit(X_tr, y_tr)

print('Accuracy of Random Forest Training: %.3f' % rf_results.best_score_)
print('The best parameters of Random Forest:', rf_results.best_params_)

rfprediction = rf_results.best_estimator_.predict(X_test)
print("Accuracy of Random Forest Testing:",metrics.accuracy_score(y_test, rfprediction))

#%% Bagging Classifier

param_grid = {
    'base_estimator__max_depth': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50, 100, 1000], # lambdas for regularization
    'max_samples': [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9], # for bootstrap sampling
    'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
 
bg = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(), n_estimators = 100, max_features = 0.5), param_grid, verbose=1, cv=LeaveOneOut(), n_jobs=-1)
bg_results = bg.fit(X_tr, y_tr)

print('Accuracy of Bagging Classifier Training: %.3f' % bg_results.best_score_)
print('The best parameters of Bagging Classifier:', bg_results.best_params_)

bgprediction = bg_results.best_estimator_.predict(X_test)
print("Accuracy of Bagging Classifier Testing:",metrics.accuracy_score(y_test, bgprediction))

#%% Ridge Classifier

param_grid = {
    'alpha' : [0.0001, 0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    'fit_intercept' : [True, False],
    'solver' : ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

rdg = GridSearchCV(estimator = RidgeClassifier(), param_grid = param_grid, verbose=1, cv=LeaveOneOut(), n_jobs=-1)
rdg_results = rdg.fit(X_tr, y_tr)

print('Accuracy of Ridge Classifier Training: %.3f' % rdg_results.best_score_)
print('The best parameters of Ridge Classifier:', rdg_results.best_params_)

rdgprediction = rdg_results.best_estimator_.predict(X_test)
print("Accuracy of Ridge Classifier Testing:",metrics.accuracy_score(y_test, rdgprediction))

#%% Accuracy of all methods


print('Accuracy of kNN Testing: %.3f' % metrics.accuracy_score(y_test, knnprediction))
print('Accuracy of Gradient Boosting Algorithm Testing: %.3f' % metrics.accuracy_score(y_test, gbcprediction))
print('Accuracy of AdaBoost Algorithm: %.3f' % metrics.accuracy_score(y_test, adaprediction))
print('Accuracy of Stochastic Gradient Descent Testing: %.3f' % metrics.accuracy_score(y_test, sgdprediction))
print('Accuracy of Support Vector Machine Testing: %.3f' % metrics.accuracy_score(y_test, svmprediction))
print('Accuracy of Decision Tree Testing: %.3f' % metrics.accuracy_score(y_test, dtprediction))
print('Accuracy of Random Forest Testing: %.3f' % metrics.accuracy_score(y_test, rfprediction))
print('Accuracy of Bagging Classifier Testing: %.3f' % metrics.accuracy_score(y_test, bgprediction))
print('Accuracy of Ridge Classifier Testing: %.3f' % metrics.accuracy_score(y_test, rdgprediction))


