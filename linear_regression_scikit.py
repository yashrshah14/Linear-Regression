import numpy as np
import csv
import sys
from sklearn import linear_model

## read csv file
f = open("./BeijingPM20100101_20151231.csv", "r")
reader = csv.reader(f)

## multiple features are set
pm = []
dp = []
humid = []
pres = []
temp = []
wind = []
precipitation = []

rownum = 0
for row in reader:
	if rownum != 0:
		pm.append((float)(row[0]))
		dp.append((float)(row[1]))
		humid.append((float)(row[2]))
		pres.append((float)(row[3]))
		temp.append((float)(row[4]))
		wind.append((float)(row[5]))
		precipitation.append((float)(row[6]))
	rownum += 1

data_len = len(pm)
num_train = (int)(data_len * 0.8)
num_test = data_len - num_train

## split dataset into two parts, train and test, validation set is automatically seperated in the scikit function when cross validation is used
X_pm_train = np.array(pm[:num_train]).reshape(num_train, 1)
X_dp_train = np.array(dp[:num_train]).reshape(num_train, 1)
X_humid_train = np.array(humid[:num_train]).reshape(num_train, 1)
X_pres_train = np.array(pres[:num_train]).reshape(num_train, 1)
X_temp_train = np.array(temp[:num_train]).reshape(num_train, 1)
X_wind_train = np.array(wind[:num_train]).reshape(num_train, 1)
Y_train = np.array(precipitation[:num_train]).reshape(num_train, 1)

def data_mean(values):
	return np.sum(values, axis = 0)/float(values.shape[0])

def data_range(values):
	return np.amax(values, axis = 0) - np.amin(values, axis = 0)

X_train = np.concatenate((X_pm_train, X_dp_train, X_humid_train, X_humid_train ** 2, X_humid_train ** 3, X_pres_train, X_pres_train ** 2, X_temp_train, X_wind_train ** 2), axis = 1)

X_train_mean = data_mean(X_train)
X_train_range = data_range(X_train)
Y_train_mean = data_mean(Y_train)
Y_train_range = data_range(Y_train)

X_train = (X_train - X_train_mean) / X_train_range
Y_train = (Y_train - Y_train_mean) / Y_train_range

X_pm_test = np.array(pm[num_train:]).reshape(num_test, 1)
X_dp_test = np.array(dp[num_train:]).reshape(num_test, 1)
X_humid_test = np.array(humid[num_train:]).reshape(num_test, 1)
X_pres_test = np.array(pres[num_train:]).reshape(num_test, 1)
X_temp_test = np.array(temp[num_train:]).reshape(num_test, 1)
X_wind_test = np.array(wind[num_train:]).reshape(num_test, 1)
Y_test = np.array(precipitation[num_train:]).reshape(num_test, 1)

X_test = np.concatenate((X_pm_test, X_dp_test, X_humid_test, X_humid_test ** 2, X_humid_test ** 3, X_pres_test, X_pres_test ** 2, X_temp_test, X_wind_test ** 2), axis = 1)

X_test = (X_test - X_train_mean) / X_train_range
Y_test = (Y_test) - Y_train_mean / Y_train_range

## use cross validation to find the optimal alpha (regularization strenth parameter)
reg = linear_model.RidgeCV(alphas = [10.0, 50.0, 100.0], fit_intercept = True, normalize = False)
reg.fit(X_train, Y_train)

Y_pred = X_train.dot(reg.coef_.T) + reg.intercept_
rmse = np.sum(np.square(Y_pred - Y_train))/(float)(Y_pred.shape[0])
print 'train rmse: %f' % rmse
print 'alpha = %f, intercept = %f, weight = ' % (reg.alpha_, reg.intercept_)
print reg.coef_
Y_pred = X_test.dot(reg.coef_.T) + reg.intercept_
rmse = np.sum(np.square(Y_pred - Y_test))/(float)(Y_pred.shape[0])
print 'test rmse: %f' % rmse
