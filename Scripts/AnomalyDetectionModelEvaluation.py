
"""
Author: Alan Danque
Date:   20210418
Purpose:Model Testing for Anomaly Detection

LinearRegression
IsolationForest
EllipticEnvelope
LocalOutlierFactor
OneClassSVM

"""

import imblearn
import seaborn as sns
import json
import pandas as pd
import glob
#from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
from pandasql import sqldf
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from numpy.core.defchararray import find
from pandas_profiling import ProfileReport
import string
from io import StringIO
from html.parser import HTMLParser
from sklearn import preprocessing
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
#use_label_encoder =False, eval_metric='logloss' - To suppress warnings.
from sklearn import metrics
from pandas import read_csv

import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import math
import os
import time
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import yaml
from pathlib import Path
import sklearn
print('scikit-learn version: {}'.format(sklearn.__version__))
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

#C:\Alan\DSC680\Project3Data
start_time = time.time()  # Overall Execution
mypath = "C:/alan/DSC680/"
base_dir = Path(mypath)
appdir = Path(os.path.dirname(base_dir))
backdir = Path(os.path.dirname(appdir))
config_path = base_dir.joinpath('Config')
yaml_filename = "config_anomaly_detection.yaml" # sys.argv[2]
ymlfile = config_path.joinpath(yaml_filename)
project_path = base_dir.joinpath('Project3Data')
data_file = project_path.joinpath('creditcard.csv')
results_path = base_dir.joinpath('Project3_Results')
results_path.mkdir(parents=True, exist_ok=True)

analytics_record = results_path.joinpath('generated_analytics_dataset.csv')
anomaly_detection_results = results_path.joinpath('anomaly_detection.csv')
pandasEDA = results_path.joinpath('Anomaly_Detection_PandasProfileReport_output.html')

# Read YAML Config
with open(ymlfile, 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
        venvpath = cfg["detection"].get("venvpath")
        ca_certs = cfg["detection"].get("ca_certs")
    except yaml.YAMLError as exc:
        print(exc)

# Read in the Telecom Churn Data
df = pd.read_csv(data_file)
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)

OrigX = X
Origy = y
OrigX_train = X_train
OrigX_test = X_test
Origy_train = y_train
Origy_test = y_test

#===========================================
# LinearRegression
# fit the model
sttime = time.time()
print("LinearRegression")
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
print(yhat)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
mse = mean_squared_error(y_test, yhat)
rmse = math.sqrt(mse)
r2score = r2_score(y_test, yhat)
print('MAE: %.3f' % mae)
print('MSE: %.3f' % mse)
print('RMSE: %.3f' % rmse)
print('r2 Score: %.3f' % r2score)
"""
# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)

# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
"""
print("LinearRegression Complete Duration: --- %s seconds ---" % (time.time() - sttime))


import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn import svm
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score
# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

# Run classifier
random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state, tol=1e-5, verbose=1, max_iter=10000))
classifier.fit(X_train, y_train)
y_score = classifier.decision_function(X_test)
average_precision = average_precision_score(y_test, y_score)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
disp = plot_precision_recall_curve(classifier, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
plt.show()


#===========================================
# IsolationForest
# evaluate model performance with outliers removed using isolation forest
# identify outliers in the training dataset
sttime = time.time()
print("IsolationForest")
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)
print(yhat)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
mse = mean_squared_error(y_test, yhat)
rmse = math.sqrt(mse)
r2score = r2_score(y_test, yhat)
print('MAE: %.3f' % mae)
print('MSE: %.3f' % mse)
print('RMSE: %.3f' % rmse)
print('r2 Score: %.3f' % r2score)
print("IsolationForest Complete Duration: --- %s seconds ---" % (time.time() - sttime))

#===========================================
# evaluate model performance with outliers removed using elliptical envelope
# identify outliers in the training dataset
sttime = time.time()
print("EllipticEnvelope")
ee = EllipticEnvelope(contamination=0.01, support_fraction=1.7)
yhat = ee.fit_predict(X_train)
print(yhat)

# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)

# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
mse = mean_squared_error(y_test, yhat)
rmse = math.sqrt(mse)
r2score = r2_score(y_test, yhat)
print('MAE: %.3f' % mae)
print('MSE: %.3f' % mse)
print('RMSE: %.3f' % rmse)
print('r2 Score: %.3f' % r2score)
print("EllipticEnvelope Complete Duration: --- %s seconds ---" % (time.time() - sttime))

#===========================================
# evaluate model performance with outliers removed using local outlier factor
# identify outliers in the training dataset
sttime = time.time()
print("LocalOutlierFactor")
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)
print(yhat)

# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
mse = mean_squared_error(y_test, yhat)
rmse = math.sqrt(mse)
r2score = r2_score(y_test, yhat)
print('MAE: %.3f' % mae)
print('MSE: %.3f' % mse)
print('RMSE: %.3f' % rmse)
print('r2 Score: %.3f' % r2score)
print("LocalOutlierFactor Complete Duration: --- %s seconds ---" % (time.time() - sttime))

#===========================================
# evaluate model performance with outliers removed using one class SVM
# identify outliers in the training dataset
sttime = time.time()
print("OneClassSVM")
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)
print(yhat)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
mse = mean_squared_error(y_test, yhat)
rmse = math.sqrt(mse)
r2score = r2_score(y_test, yhat)
print('MAE: %.3f' % mae)
print('MSE: %.3f' % mse)
print('RMSE: %.3f' % rmse)
print('r2 Score: %.3f' % r2score)
print("OneClassSVM Complete Duration: --- %s seconds ---" % (time.time() - sttime))

#===========================================

print("Complete Duration: --- %s seconds ---" % (time.time() - start_time))


import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn import svm
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

random_state = np.random.RandomState(0)
# Run classifier
classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
classifier.fit(X_train, y_train)
y_score = classifier.decision_function(X_test)

disp = plot_precision_recall_curve(classifier, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))

