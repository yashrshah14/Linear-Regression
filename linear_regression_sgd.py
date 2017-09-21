import numpy as np
import csv
import sys
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

X_list = np.split(X_train, [(int)(num_train * 0.8)])
X_train = X_list[0]
X_validate = X_list[1]
Y_list = np.split(Y_train, [(int)(num_train * 0.8)])
Y_train = Y_list[0]
Y_validate = Y_list[1]

X_train = (X_train - X_train_mean) / X_train_range
X_validate = (X_validate - X_train_mean) / X_train_range
Y_train = (Y_train - Y_train_mean) / Y_train_range
Y_validate = (Y_validate - Y_train_mean) / Y_train_range

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

## initialize weight matrix and bias
b = np.random.rand(1,9)
b_bias = 0.01
n_epoch = 100
l_rate = 0.1
for epoch in range(n_epoch):
	for row in range(X_train.shape[0]):
		Y_pred = X_train[row,:].dot(b.T) + b_bias
		err = Y_pred - Y_train[row]
		b -= l_rate * err * X_train[row,:]
		b_bias -= l_rate * err
	Y_pred = X_validate.dot(b.T) + b_bias
	rmse = np.sum(np.square(Y_validate - Y_pred))/(float)(Y_pred.shape[0])
	print("epoch %d, rmse %f" % (epoch, rmse))

Y_pred = X_test.dot(b.T) + b_bias
rmse = np.sum(np.square(Y_test - Y_pred))/(float)(Y_test.shape[0])
print 'alpha = %f, intercept = %f, weight = ' % (l_rate, b_bias)
print b
print "test rmse: %f" % rmse
