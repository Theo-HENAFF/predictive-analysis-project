"""
=============================================
Comparison of kernel ridge regression and SVR
=============================================

Both kernel ridge regression (KRR) and SVR learn a non-linear function by
employing the kernel trick, i.e., they learn a linear function in the space
induced by the respective kernel which corresponds to a non-linear function in
the original space. They differ in the loss functions (ridge versus
epsilon-insensitive loss). In contrast to SVR, fitting a KRR can be done in
closed-form and is typically faster for medium-sized datasets. On the other
hand, the learned model is non-sparse and thus slower than SVR at
prediction-time.

This example illustrates both methods on an artificial dataset, which
consists of a sinusoidal target function and strong noise added to every fifth
datapoint. The first figure compares the learned model of KRR and SVR when both
complexity/regularization and bandwidth of the RBF kernel are optimized using
grid-search. The learned functions are very similar; however, fitting KRR is
approx. seven times faster than fitting SVR (both with grid-search). However,
prediction of 100000 target values is more than tree times faster with SVR
since it has learned a sparse model using only approx. 1/3 of the 100 training
datapoints as support vectors.

The next figure compares the time for fitting and prediction of KRR and SVR for
different sizes of the training set. Fitting KRR is faster than SVR for medium-
sized training sets (less than 1000 samples); however, for larger training sets
SVR scales better. With regard to prediction time, SVR is faster than
KRR for all sizes of the training set because of the learned sparse
solution. Note that the degree of sparsity and thus the prediction time depends
on the parameters epsilon and C of the SVR.
"""

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause


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

X_plot = np.linspace(0, len(y), N)[:, None]


## #############################################################################
## Fit regression model
train_size = 30
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": [1e-3, 2]})

#kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
#                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
#                              "gamma": np.logspace(-2, 2, 5)})

t0 = time.time()
svr.fit(X[:train_size], y[:train_size])
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
plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors',
            zorder=2, edgecolors=(0, 0, 0))
plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1,
            edgecolors=(0, 0, 0))
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
