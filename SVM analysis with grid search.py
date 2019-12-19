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

#from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


# #############################################################################
# Generate sample data
a = []
line_count = 0
train_size = 40

with open('indice_data_FRA.csv') as csv_file:
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

N=len(a)

new_X = np.linspace(0,N-1,train_size, dtype=int)
new_X=np.ndarray.tolist(new_X)
temp =[]
for i in new_X:
    temp.append([i])
new_X = temp
X_plot = np.linspace(0, N-1, N)[:, None]


## #############################################################################
## Fit regression model
svr = GridSearchCV(SVR(kernel='rbf', gamma= 0.8), cv=8,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-1.2, 1.8, 7)})

t0 = time.time()
svr.fit(new_X[:train_size], y[:train_size])
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

t0 = time.time()

sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("Support vector ratio: %.3f" % sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s"
      % (X_plot.shape[0], svr_predict))

t0 = time.time()


# #############################################################################
# Look at the results
new_y=[]
for i in new_X:
    new_y.append(y[i[0]])



sv_ind = svr.best_estimator_.support_
plt.scatter(np.array(new_X), np.array(new_y), c='r', s=50, label='SVR support vectors',
            zorder=2, edgecolors=(0, 0, 0))

plt.plot(y,c='b')
plt.plot(X_plot, y_svr, c='r',
         label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
plt.title('SVR')
plt.legend()

plt.show()



# #############################################################################
# Accuracy, precision score
#print('Accuracy Score : ' + str(accuracy_score(y,y_svr[:206])))
#print('Precision Score : ' + str(precision_score(y,y_svr[:206])))
#print('Recall Score : ' + str(recall_score(y,y_svr[:206])))
#print('F1 Score : ' + str(f1_score(y,y_svr)))
