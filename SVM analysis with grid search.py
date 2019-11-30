# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 8:10:59 2019

@author: Théo
"""
#Inspired from Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>


import time
import csv
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

rng = np.random.RandomState(0)

# #############################################################################
# Generate sample data
a = []
line_count = 0
N=206
train_size = 30

with open('data_KOR.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else :
            temp = row[6][0:len(row[6])-1]
            if temp[0] == '.':
                temp = '0'+temp
            elif temp[0] == '-' and temp[1] == '.':
                temp = '-0'+ temp[1:]
            a.append(float(temp))
y = np.array(a)
X = len(y) * rng.rand(300, 1)

new_X = np.linspace(0,N,train_size)

X_plot = np.linspace(0, len(y), N)[:, None]


## #############################################################################
## Fit regression model
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": [1e-3, 2]})

#kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
#                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
#                              "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(new_X, y[:train_size]) #old_X = X[:train_size]
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

t0 = time.time()
#kr.fit(X[:train_size], y[:train_size])
#kr_fit = time.time() - t0
#print("KRR complexity and bandwidth selected and model fitted in %.3f s"
#      % kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("Support vector ratio: %.3f" % sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s"
      % (X_plot.shape[0], svr_predict))

t0 = time.time()
#y_kr = kr.predict(X_plot)
#kr_predict = time.time() - t0
#print("KRR prediction for %d inputs in %.3f s"
#      % (X_plot.shape[0], kr_predict))
#

# #############################################################################
# Look at the results
sv_ind = svr.best_estimator_.support_
plt.scatter(new_X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors',
            zorder=2, edgecolors=(0, 0, 0)) #old_X = X[sv_ind]
plt.scatter(new_X[:100], y[:100], c='k', label='data', zorder=1,
            edgecolors=(0, 0, 0)) #old_X = X[:100]
plt.plot(y,c='b')
plt.plot(X_plot, y_svr, c='r',
         label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
#plt.plot(X_plot, y_kr, c='g',
#         label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
plt.title('SVR')
plt.legend()

plt.show()



# #############################################################################
# Accuracy, precision score
print('Accuracy Score : ' + str(accuracy_score(y,y_svr[:206])))
print('Precision Score : ' + str(precision_score(y,y_svr[:206])))
print('Recall Score : ' + str(recall_score(y,y_svr[:206])))
print('F1 Score : ' + str(f1_score(y,y_svr)))
